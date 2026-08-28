import io
import json
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from analysis import PipelineParams, load_grayscale, preprocess, analyze
from calibration import (
    CalibrationProfile,
    profiles_to_json,
    profiles_from_json,
    find_matching_profile,
    scaled_um_per_px,
    um_per_px_from_two_points,
    um_per_px_from_rectangle,
)

st.set_page_config(page_title="Analiza korneal. endotela", layout="wide")

st.title("🔬 Analiza slike korneal. endotela (slit-lamp / specular)")

with st.expander("ℹ️ O aplikaciji, kalibraciji i ograničenjima", expanded=False):
    st.markdown(
        """
Ova aplikacija je **istraživački / edukativni prototip**, inspiriran pristupom
opisanim u radu *"Low-Cost, Smartphone-Based Specular Imaging and Automated
Analysis of the Corneal Endothelium"* (Translational Vision Science &
Technology, 2021,
[PMC8024782](https://pmc.ncbi.nlm.nih.gov/articles/PMC8024782/)) — smartphone
slit-lamp specular fotografija + automatska segmentacija stanica endotela.
Koristi se standardna, otvoreno dokumentirana klasična tehnika segmentacije
("marker-controlled watershed"), a ne proizvođačev točan (patentirani) algoritam.

**O kalibraciji (µm/piksel):** tražio sam javno dostupne specifikacije
(rezolucija senzora, vidno polje po povećanju) za CSO-ovu kameru na
biomikroskopu i nisam uspio pronaći nikakav zvanični, objavljeni podatak o
tome koliko mm pokriva kadar pri pojedinom povećanju — CSO to ne objavljuje
javno. Zato aplikacija **ne pretpostavlja** tu vrijednost umjesto tebe, nego
nudi dva načina da je dobiješ i spremiš kao profil za buduće slike:

1. **Ako već znaš** vidno polje (mm) ili µm/px za pojedino povećanje (npr. iz
   dokumentacije uređaja ili prijašnje kalibracije) — unesi ga ručno u
   izborniku "Kalibracijski profili".
2. **Ako ne znaš** — koristi ugrađeni alat "🔧 Kalibracija ravnalom" ispod:
   snimi mjernu pločicu (npr. predmetno staklo s mikrometarskom skalom, ili
   bilo koji ravnalo/objekt poznate veličine) kroz isti okular/kameru pri
   istom povećanju, povuci liniju preko poznate udaljenosti na slici, i
   aplikacija izračuna µm/px za tu kombinaciju povećanja + rezolucije. To
   samo trebaš napraviti **jednom po povećanju** — profil se sprema i
   prepoznaje automatski kod idućih slika iste rezolucije.

**Zašto crop ne kvari kalibraciju:** kalibracija (µm/px) je svojstvo
kamere+optike+povećanja, ne onoga što naknadno izrežeš iz slike. Sve dok se
slika ne smanjuje/povećava (samo se bira pravokutnik piksela), µm/px ostaje
isti prije i poslije cropa — zato crop alat ispod samo bira dio slike za
analizu, a kalibracija se prenosi nepromijenjena.

**Ostala ograničenja:**
- Kvaliteta segmentacije ovisi o kontrastu/oštrini slike i o parametrima
  koje namjestite.
- Broj "stranica" stanice (za HEX%) procjenjuje se preko broja susjednih
  stanica u mozaiku — praktična zamjena za "triple-point" metodu iz literature.
- Ovo **nije klinički validiran medicinski uređaj**.
        """
    )

if "calib_profiles" not in st.session_state:
    st.session_state.calib_profiles = []


def _profile_by_label(label):
    for p in st.session_state.calib_profiles:
        if p.label() == label:
            return p
    return None


def _scale_drag_to_native(drag: dict, native_w: int, native_h: int):
    """streamlit_image_coordinates returns click/drag coords in the
    *displayed* (possibly resized) image's pixel space, plus the displayed
    element's width/height -- convert back to the original image's pixel
    coordinates."""
    disp_w = drag.get("width") or native_w
    disp_h = drag.get("height") or native_h
    sx = native_w / disp_w
    sy = native_h / disp_h
    x1 = drag["x1"] * sx
    y1 = drag["y1"] * sy
    x2 = drag["x2"] * sx
    y2 = drag["y2"] * sy
    return x1, y1, x2, y2


def _scale_click_to_native(click: dict, native_w: int, native_h: int):
    """Same idea as _scale_drag_to_native, for a plain (non-drag) single
    click event, whose keys are 'x'/'y' rather than 'x1'/'y1'/'x2'/'y2'."""
    disp_w = click.get("width") or native_w
    disp_h = click.get("height") or native_h
    sx = native_w / disp_w
    sy = native_h / disp_h
    return click["x"] * sx, click["y"] * sy


# ---------------------------------------------------------------- sidebar --
st.sidebar.header("1. Učitaj sliku")
uploaded = st.sidebar.file_uploader(
    "Slika endotela (specular / slit-lamp)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
)

use_demo = False
if uploaded is None:
    use_demo = st.sidebar.checkbox("Koristi sintetsku demo sliku", value=True)


def make_demo_image(size=480, n_cells=140, seed=7, lloyd_iters=4):
    """Synthetic Voronoi mosaic (Lloyd-relaxed so cells are fairly regular
    and hexagon-heavy, like real endothelium) that mimics a specular
    photo, purely so the app has something to show without a real image."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, size, size=(n_cells, 2))
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)

    for _ in range(lloyd_iters):
        tree = cKDTree(pts)
        _, idx = tree.query(grid)
        pts = np.array([
            grid[idx == i].mean(axis=0) if np.any(idx == i) else pts[i]
            for i in range(n_cells)
        ])

    tree = cKDTree(pts)
    d, idx = tree.query(grid, k=2)
    idx_map = idx[:, 0].reshape(size, size)

    right_diff = np.zeros_like(idx_map, dtype=bool)
    right_diff[:, :-1] = idx_map[:, :-1] != idx_map[:, 1:]
    down_diff = np.zeros_like(idx_map, dtype=bool)
    down_diff[:-1, :] = idx_map[:-1, :] != idx_map[1:, :]
    border_mask = right_diff | down_diff

    img = np.full((size, size), 60, dtype=np.float32)  # dark cell body
    img[border_mask] = 220.0  # bright specular border
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.7)

    noise = rng.normal(0, 8, size=(size, size))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # vignette to mimic uneven slit-lamp illumination
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (size / 1.4)
    vign = np.clip(1 - 0.5 * r, 0.4, 1.0)
    img = np.clip(img * vign, 0, 255).astype(np.uint8)
    return img


# ------------------------------------------------------------- load image --
gray_full = None
if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_arr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img_arr is None:
        st.error("Nije moguće učitati sliku. Provjerite format datoteke.")
    else:
        if img_arr.ndim == 3:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        gray_full = load_grayscale(img_arr)
elif use_demo:
    gray_full = make_demo_image()

if gray_full is None:
    st.warning("Učitajte sliku endotela u lijevom izborniku da biste započeli analizu.")
    st.stop()

native_h, native_w = gray_full.shape[:2]

# ------------------------------------------------------- 2. calibration UI --
st.sidebar.header("2. Kalibracijski profili")

# The "add/import/delete profiles" controls are handled FIRST, before the
# selectbox/auto-match logic below reads st.session_state.calib_profiles --
# so a profile added (or deleted) this run is immediately reflected in the
# dropdown in the same pass, with no need for an explicit st.rerun() (which
# would risk resetting other widgets' state further down the script -- see
# the note by the crop tool).
with st.sidebar.expander("➕ Dodaj / uredi kalibracijske profile"):
    st.caption("Dodaj profil ako već znaš µm/px ili vidno polje (FOV) za neko "
               "povećanje. Ako ne znaš, koristi alat 'Kalibracija ravnalom' "
               "ispod glavne slike — on može spremiti profil umjesto tebe.")
    new_name = st.text_input("Naziv (npr. '40x')", key="new_profile_name")
    c1, c2 = st.columns(2)
    new_w = c1.number_input("Nativna širina (px)", min_value=1, value=native_w, key="new_profile_w")
    new_h = c2.number_input("Nativna visina (px)", min_value=1, value=native_h, key="new_profile_h")
    mode = st.radio("Unos kalibracije preko:", ["µm/px izravno", "širina vidnog polja (mm)"], key="new_profile_mode")
    if mode == "µm/px izravno":
        new_umpx = st.number_input("µm/px", min_value=0.001, value=1.0, step=0.001, format="%.4f", key="new_profile_umpx")
    else:
        fov_mm = st.number_input("Širina vidnog polja (mm)", min_value=0.01, value=5.0, step=0.01, key="new_profile_fov")
        new_umpx = (fov_mm * 1000.0) / new_w
        st.caption(f"= {new_umpx:.4f} µm/px")
    if st.button("Dodaj profil", key="add_profile_btn"):
        if new_name.strip():
            st.session_state.calib_profiles.append(
                CalibrationProfile(name=new_name.strip(), native_w=int(new_w), native_h=int(new_h), um_per_px=float(new_umpx))
            )
            # No explicit st.rerun() here: the button click itself already
            # triggers Streamlit's normal automatic rerun. Calling rerun()
            # again would abort this script partway through -- before it
            # reaches the crop-tool widgets further down -- which makes
            # Streamlit treat their session_state as orphaned/stale and
            # reset them to their bare defaults (e.g. min_value) on the
            # next run. The list mutation above is enough; it's already
            # picked up on the automatic rerun.
        else:
            st.error("Unesi naziv profila.")

    if st.session_state.calib_profiles:
        st.markdown("**Spremljeni profili:**")
        for i, p in enumerate(st.session_state.calib_profiles):
            pc1, pc2 = st.columns([5, 1])
            pc1.write(p.label())
            if pc2.button("🗑️", key=f"del_profile_{i}"):
                st.session_state.calib_profiles.pop(i)

        export_json = profiles_to_json(st.session_state.calib_profiles)
        st.download_button("⬇️ Izvezi profile (JSON)", export_json, "kalibracijski_profili.json", "application/json")

    imported = st.file_uploader("⬆️ Uvezi profile (JSON)", type=["json"], key="import_profiles")
    if imported is not None:
        try:
            loaded = profiles_from_json(imported.read().decode("utf-8"))
            existing_names = {p.name for p in st.session_state.calib_profiles}
            added = 0
            for p in loaded:
                if p.name not in existing_names:
                    st.session_state.calib_profiles.append(p)
                    added += 1
            st.success(f"Uvezeno {added} novih profila.")
        except Exception as e:
            st.error(f"Neispravan JSON: {e}")

matched_profile = find_matching_profile(st.session_state.calib_profiles, native_w, native_h)

profile_labels = ["(ručni unos µm/px)"] + [p.label() for p in st.session_state.calib_profiles]
default_idx = 0
if matched_profile is not None:
    default_idx = profile_labels.index(matched_profile.label())

chosen_label = st.sidebar.selectbox(
    "Aktivni profil", profile_labels, index=default_idx,
    help="Profili se pamte samo tijekom ove sesije preglednika. Izvezi ih kao "
         "JSON da ih sačuvaš i uveziš iduci put."
)
chosen_profile = _profile_by_label(chosen_label)

resolved_um_per_px = 1.0
if chosen_profile is not None:
    if chosen_profile.native_w == native_w and chosen_profile.native_h == native_h:
        st.sidebar.success(
            f"✅ Rezolucija slike odgovara profilu '{chosen_profile.name}' "
            f"({native_w}x{native_h} px) — kalibracija primijenjena automatski."
        )
        resolved_um_per_px = chosen_profile.um_per_px
    else:
        st.sidebar.warning(
            f"Rezolucija ove slike ({native_w}x{native_h} px) ne odgovara "
            f"profilu '{chosen_profile.name}' ({chosen_profile.native_w}x"
            f"{chosen_profile.native_h} px)."
        )
        assume_full_frame = st.sidebar.checkbox(
            "Ipak je ovo cijeli izvorni kadar (samo drugačije spremljene "
            "rezolucije, nije prethodno rezan)",
            value=False,
        )
        if assume_full_frame:
            resolved_um_per_px = scaled_um_per_px(chosen_profile, native_w, native_h)
            st.sidebar.info(f"Preračunata kalibracija: {resolved_um_per_px:.4f} µm/px")
        else:
            resolved_um_per_px = 1.0

um_per_px = st.sidebar.number_input(
    "Mikrometara po pikselu (µm/px) — konačna vrijednost",
    min_value=0.001, max_value=200.0, value=float(resolved_um_per_px), step=0.001,
    format="%.4f",
    help="Automatski predložena vrijednost iz odabranog profila, ali uvijek "
         "možeš ručno prepisati."
)

# ------------------------------------------------- ruler calibration tool --
with st.expander("🔧 Kalibracija ravnalom / mjernom pločicom (ako ne znaš µm/px)"):
    st.markdown(
        "Snimi mjernu pločicu ili milimetarski papir kroz **isti okular i "
        "kameru, pri istom povećanju** kao slike koje analiziraš (isto što si "
        "dosad radio/la u Phoenixu). Učitaj tu sliku i izaberi način mjerenja "
        "ispod."
    )
    calib_img_file = st.file_uploader(
        "Slika mjerne pločice / mm papira", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="calib_ruler_img"
    )
    if calib_img_file is not None:
        cb = np.frombuffer(calib_img_file.read(), np.uint8)
        calib_arr = cv2.imdecode(cb, cv2.IMREAD_UNCHANGED)
        if calib_arr is not None:
            if calib_arr.ndim == 3:
                calib_arr = cv2.cvtColor(calib_arr, cv2.COLOR_BGR2RGB)
            calib_h, calib_w = calib_arr.shape[:2]
            disp_width = min(800, calib_w)

            calib_mode = st.radio(
                "Način mjerenja",
                ["Linija (2 točke)", "Pravokutnik / mreža (4 kuta — preciznije, kao u Phoenixu)"],
                key="calib_mode",
                horizontal=True,
            )

            if calib_mode == "Linija (2 točke)":
                st.caption("Povuci (klikni i povuci) liniju preko poznate udaljenosti:")
                drag = streamlit_image_coordinates(
                    calib_arr, width=disp_width, key="ruler_drag", click_and_drag=True
                )

                if drag is not None and drag != st.session_state.get("_last_ruler_drag"):
                    # No st.rerun(): the component's value changing already
                    # triggered this run automatically, and session_state set
                    # here is picked up by the number_inputs created just
                    # below in this same pass. An extra rerun() would abort
                    # before reaching later widgets and reset their state
                    # (see the note by the crop tool for the full story).
                    st.session_state["_last_ruler_drag"] = drag
                    x1, y1, x2, y2 = _scale_drag_to_native(drag, calib_w, calib_h)
                    st.session_state["ruler_x1"] = float(x1)
                    st.session_state["ruler_y1"] = float(y1)
                    st.session_state["ruler_x2"] = float(x2)
                    st.session_state["ruler_y2"] = float(y2)

                st.session_state.setdefault("ruler_x1", 0.0)
                st.session_state.setdefault("ruler_y1", 0.0)
                st.session_state.setdefault("ruler_x2", float(calib_w))
                st.session_state.setdefault("ruler_y2", 0.0)

                rc1, rc2, rc3, rc4 = st.columns(4)
                rx1 = rc1.number_input("x1 (px)", key="ruler_x1")
                ry1 = rc2.number_input("y1 (px)", key="ruler_y1")
                rx2 = rc3.number_input("x2 (px)", key="ruler_x2")
                ry2 = rc4.number_input("y2 (px)", key="ruler_y2")

                preview = cv2.cvtColor(
                    calib_arr if calib_arr.ndim == 2 else cv2.cvtColor(calib_arr, cv2.COLOR_RGB2GRAY),
                    cv2.COLOR_GRAY2RGB,
                )
                cv2.line(preview, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255, 60, 60), max(1, calib_w // 300))
                st.image(preview, caption="Pregled linije mjerenja", width=disp_width)

                known_mm = st.number_input("Stvarna udaljenost između točaka (mm)", min_value=0.001, value=1.0, step=0.001, format="%.4f")
                try:
                    computed_umpx = um_per_px_from_two_points(rx1, ry1, rx2, ry2, known_mm)
                    st.success(f"Izračunata kalibracija: **{computed_umpx:.4f} µm/px** "
                               f"(za rezoluciju {calib_w}x{calib_h} px)")

                    save_name = st.text_input("Naziv profila za spremanje (npr. '40x')", key="ruler_save_name")
                    if st.button("💾 Spremi kao kalibracijski profil", key="save_line_profile_btn"):
                        if save_name.strip():
                            st.session_state.calib_profiles.append(
                                CalibrationProfile(name=save_name.strip(), native_w=calib_w, native_h=calib_h, um_per_px=computed_umpx)
                            )
                            st.success(f"Profil '{save_name.strip()}' spremljen — odaberi ga gore za sliku iste rezolucije.")
                        else:
                            st.error("Unesi naziv profila.")
                except ValueError as e:
                    st.error(str(e))

            else:
                # ---- 4-corner rectangle/grid calibration (mirrors the
                # Phoenix workflow: stretch a known rectangle/grid to fit
                # graph paper). Uses all 4 sides instead of one line, which
                # averages out click-precision error over a larger
                # baseline, and cross-checks the x-axis vs y-axis scale.
                st.caption(
                    "Klikni redom **4 kuta** poznatog pravokutnika na mm papiru "
                    "(npr. kut jednog ili više kvadratića): gore-lijevo → "
                    "gore-desno → dolje-desno → dolje-lijevo."
                )

                corner_labels = ["Gore-lijevo", "Gore-desno", "Dolje-desno", "Dolje-lijevo"]
                default_corners = [(0.0, 0.0), (float(calib_w), 0.0),
                                    (float(calib_w), float(calib_h)), (0.0, float(calib_h))]
                for i in range(4):
                    st.session_state.setdefault(f"corner_{i}_x", default_corners[i][0])
                    st.session_state.setdefault(f"corner_{i}_y", default_corners[i][1])
                st.session_state.setdefault("corner_count", 0)

                next_idx = st.session_state["corner_count"]
                if next_idx < 4:
                    st.info(f"Sljedeći klik postavlja kut: **{corner_labels[next_idx]}**")
                else:
                    st.success("Sva 4 kuta postavljena — fino podesi brojevima ispod ako treba, ili resetiraj.")

                click = streamlit_image_coordinates(calib_arr, width=disp_width, key="corner_click")
                if click is not None and click != st.session_state.get("_last_corner_click"):
                    st.session_state["_last_corner_click"] = click
                    if st.session_state["corner_count"] < 4:
                        cx, cy = _scale_click_to_native(click, calib_w, calib_h)
                        idx = st.session_state["corner_count"]
                        st.session_state[f"corner_{idx}_x"] = float(cx)
                        st.session_state[f"corner_{idx}_y"] = float(cy)
                        st.session_state["corner_count"] = idx + 1

                def _reset_corners():
                    st.session_state["corner_count"] = 0
                    for i in range(4):
                        st.session_state[f"corner_{i}_x"] = default_corners[i][0]
                        st.session_state[f"corner_{i}_y"] = default_corners[i][1]

                st.button("🔄 Resetiraj kutove", on_click=_reset_corners, key="reset_corners_btn")

                corners = []
                for i in range(4):
                    ccol1, ccol2 = st.columns(2)
                    cxv = ccol1.number_input(f"{corner_labels[i]} — x (px)", key=f"corner_{i}_x")
                    cyv = ccol2.number_input(f"{corner_labels[i]} — y (px)", key=f"corner_{i}_y")
                    corners.append((cxv, cyv))

                preview = cv2.cvtColor(
                    calib_arr if calib_arr.ndim == 2 else cv2.cvtColor(calib_arr, cv2.COLOR_RGB2GRAY),
                    cv2.COLOR_GRAY2RGB,
                )
                pts = np.array([[int(x), int(y)] for x, y in corners], dtype=np.int32)
                cv2.polylines(preview, [pts], isClosed=True, color=(255, 60, 60), thickness=max(1, calib_w // 300))
                for i, (x, y) in enumerate(corners):
                    cv2.circle(preview, (int(x), int(y)), max(3, calib_w // 150), (60, 140, 220), -1)
                st.image(preview, caption="Pregled pravokutnika mjerenja", width=disp_width)

                wc1, wc2 = st.columns(2)
                rect_w_mm = wc1.number_input("Širina pravokutnika (mm)", min_value=0.001, value=1.0, step=0.001, format="%.4f", key="rect_w_mm")
                rect_h_mm = wc2.number_input("Visina pravokutnika (mm)", min_value=0.001, value=1.0, step=0.001, format="%.4f", key="rect_h_mm")

                try:
                    result = um_per_px_from_rectangle(corners, rect_w_mm, rect_h_mm)
                    st.success(f"Izračunata kalibracija: **{result.um_per_px_avg:.4f} µm/px** "
                               f"(za rezoluciju {calib_w}x{calib_h} px)")
                    st.caption(
                        f"Po X-osi: {result.um_per_px_x:.4f} µm/px · Po Y-osi: {result.um_per_px_y:.4f} µm/px "
                        f"· razlika: {result.mismatch_percent:.1f}%"
                    )
                    if result.mismatch_percent > 5:
                        st.warning(
                            "Razlika između X i Y osi je preko 5% — provjeri jesu li kutovi točno "
                            "postavljeni na stvarne kutove pravokutnika prije spremanja."
                        )

                    save_name2 = st.text_input("Naziv profila za spremanje (npr. '40x')", key="rect_save_name")
                    if st.button("💾 Spremi kao kalibracijski profil", key="save_rect_profile_btn"):
                        if save_name2.strip():
                            st.session_state.calib_profiles.append(
                                CalibrationProfile(name=save_name2.strip(), native_w=calib_w, native_h=calib_h, um_per_px=result.um_per_px_avg)
                            )
                            st.success(f"Profil '{save_name2.strip()}' spremljen — odaberi ga gore za sliku iste rezolucije.")
                        else:
                            st.error("Unesi naziv profila.")
                except ValueError as e:
                    st.error(str(e))
        else:
            st.error("Nije moguće učitati sliku mjerne pločice.")

# ------------------------------------------------------------ crop tool --
st.subheader("✂️ Odaberi dio slike za analizu")
phys_w_mm = native_w * um_per_px / 1000.0
phys_h_mm = native_h * um_per_px / 1000.0
st.caption(
    f"Puna slika: {native_w}x{native_h} px ≈ {phys_w_mm:.2f} x {phys_h_mm:.2f} mm "
    f"(pri {um_per_px:.4f} µm/px). Crop ne mijenja kalibraciju — samo bira "
    "područje za analizu (npr. da izbjegneš neoštre rubove ili artefakte)."
)

# (Re)initialize crop bounds whenever a differently-sized image shows up
# (new upload, or first run) so widget state always matches valid bounds.
if st.session_state.get("_crop_native_size") != (native_w, native_h):
    st.session_state["_crop_native_size"] = (native_w, native_h)
    st.session_state["crop_left"] = 0
    st.session_state["crop_top"] = 0
    st.session_state["crop_right"] = native_w
    st.session_state["crop_bottom"] = native_h

disp_width = min(800, native_w)
crop_drag = streamlit_image_coordinates(
    gray_full, width=disp_width, key="crop_drag", click_and_drag=True
)
if crop_drag is not None and crop_drag != st.session_state.get("_last_crop_drag"):
    # No st.rerun() here -- see the note on the ruler tool's drag handler
    # above for why: the component value change already triggered this
    # run, and the crop_* number_inputs created just below in this same
    # pass will pick up the values set here directly.
    st.session_state["_last_crop_drag"] = crop_drag
    x1, y1, x2, y2 = _scale_drag_to_native(crop_drag, native_w, native_h)
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])
    st.session_state["crop_left"] = int(max(0, left))
    st.session_state["crop_top"] = int(max(0, top))
    st.session_state["crop_right"] = int(min(native_w, right))
    st.session_state["crop_bottom"] = int(min(native_h, bottom))

def _reset_crop():
    # Runs as an on_click callback, i.e. BEFORE the script body below
    # (which instantiates the crop_* widgets) re-executes -- mutating
    # session_state here is safe, whereas doing it after those widgets
    # have already been created in this same run would raise a
    # StreamlitAPIException.
    st.session_state["crop_left"] = 0
    st.session_state["crop_top"] = 0
    st.session_state["crop_right"] = native_w
    st.session_state["crop_bottom"] = native_h


cc1, cc2, cc3, cc4, cc5 = st.columns([1, 1, 1, 1, 1])
crop_left = cc1.number_input("Lijevo (px)", min_value=0, max_value=native_w - 1, key="crop_left")
crop_top = cc2.number_input("Gore (px)", min_value=0, max_value=native_h - 1, key="crop_top")
crop_right = cc3.number_input("Desno (px)", min_value=1, max_value=native_w, key="crop_right")
crop_bottom = cc4.number_input("Dolje (px)", min_value=1, max_value=native_h, key="crop_bottom")
cc5.button("↺ Cijela slika", on_click=_reset_crop)

crop_left, crop_right = sorted([int(crop_left), int(crop_right)])
crop_top, crop_bottom = sorted([int(crop_top), int(crop_bottom)])
crop_right = max(crop_right, crop_left + 1)
crop_bottom = max(crop_bottom, crop_top + 1)

preview_full = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2RGB)
cv2.rectangle(preview_full, (crop_left, crop_top), (crop_right, crop_bottom), (255, 60, 60), max(1, native_w // 300))
st.image(preview_full, caption="Pregled odabranog područja (crveni okvir) — povuci gore ili uredi brojeve", width=disp_width)

gray = gray_full[crop_top:crop_bottom, crop_left:crop_right]
crop_w_mm = (crop_right - crop_left) * um_per_px / 1000.0
crop_h_mm = (crop_bottom - crop_top) * um_per_px / 1000.0
st.caption(f"Odabrano područje za analizu: {crop_right - crop_left}x{crop_bottom - crop_top} px "
           f"≈ {crop_w_mm:.2f} x {crop_h_mm:.2f} mm")

if use_demo and uploaded is None:
    st.info("Prikazana je **sintetska demo slika** (nije stvarni pacijent) "
            "samo da vidite kako alat radi.")

# --------------------------------------------------- 3. processing params --
st.sidebar.header("3. Parametri obrade slike")
invert = st.sidebar.checkbox(
    "Invertiraj (stanice svijetle / granice tamne)", value=False
)
clahe_clip = st.sidebar.slider("CLAHE kontrast (clip limit)", 0.5, 8.0, 2.0, 0.1)
gaussian_sigma = st.sidebar.slider("Predglačanje (Gaussian sigma)", 0.0, 3.0, 1.0, 0.1)
adaptive_block = st.sidebar.slider(
    "Zaglađivanje prije praga (utječe na debljinu granica)", 11, 121, 41, 2
)
adaptive_c = st.sidebar.slider(
    "Pomak praga (+ = manje stanica, - = više stanica/šuma)", -10, 10, 0, 1
)
min_cell_px = st.sidebar.slider("Min. veličina stanice (px, šum)", 5, 300, 25, 1)
min_marker_distance = st.sidebar.slider("Min. razmak markera (px)", 1, 30, 6, 1)
exclude_border = st.sidebar.checkbox("Isključi stanice na rubu slike", value=True)

st.sidebar.header("4. Prikaz")
show_side_colors = st.sidebar.checkbox("Oboji stanice po broju stranica", value=True)

params = PipelineParams(
    clahe_clip=clahe_clip,
    gaussian_sigma=gaussian_sigma,
    invert=invert,
    adaptive_block=adaptive_block,
    adaptive_c=adaptive_c,
    min_cell_px=min_cell_px,
    min_marker_distance=min_marker_distance,
    exclude_border_cells=exclude_border,
    um_per_px=um_per_px,
)

with st.spinner("Obrada slike i segmentacija stanica..."):
    enhanced = preprocess(gray, params)
    stats = analyze(gray, params)

# ------------------------------------------------------------------ views --
col1, col2, col3 = st.columns(3)
col1.image(gray, caption="Odabrano područje (grayscale)", width="stretch", clamp=True)
col2.image(enhanced, caption="Nakon predobrade (CLAHE + normalizacija)", width="stretch", clamp=True)

overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
overlay[stats.boundaries] = [255, 60, 60]
col3.image(overlay, caption="Segmentirane granice stanica", width="stretch")

st.subheader("📊 Morfometrijski parametri")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Broj analiziranih stanica", f"{stats.n_cells}")
m2.metric("ECD (stanica/mm²)", f"{stats.ecd:,.0f}")
m3.metric("Prosj. površina (µm²)", f"{stats.mean_area:,.1f}")
m4.metric("CV % (polimegatizam)", f"{stats.cv_percent:,.1f}%")
m5.metric("HEX % (heksagonalnost)", f"{stats.hex_percent:,.1f}%")

if chosen_profile is None:
    st.caption("⚠️ Nema aktivnog kalibracijskog profila — ECD/površine koriste "
               f"ručno upisanu vrijednost ({um_per_px:.4f} µm/px). Za točne "
               "apsolutne vrijednosti postavi profil ili koristi alat za "
               "kalibraciju ravnalom iznad.")

if stats.n_cells < 50:
    st.warning(
        f"Analizirano je samo **{stats.n_cells}** stanica. Pouzdana klinička "
        "morfometrija endotela obično zahtijeva ≥ 75–100 stanica — smatrajte "
        "ove brojke orijentacijskima i pokušajte poboljšati kontrast/kvalitetu "
        "slike, prilagoditi parametre segmentacije, ili odabrati veće područje."
    )

st.divider()

colA, colB = st.columns(2)

with colA:
    st.markdown("**Raspodjela površina stanica**")
    fig, ax = plt.subplots()
    if stats.n_cells > 0:
        ax.hist(stats.areas_um2, bins=min(20, max(5, stats.n_cells // 3)), color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Površina stanice (µm²)")
    ax.set_ylabel("Broj stanica")
    st.pyplot(fig, width="stretch")

with colB:
    st.markdown("**Raspodjela broja stranica (pleomorfizam)**")
    fig2, ax2 = plt.subplots()
    if stats.n_cells > 0:
        vals, counts = np.unique(stats.sides, return_counts=True)
        ax2.bar(vals, counts, color="#F58518")
        ax2.set_xlabel("Broj stranica poligona stanice")
        ax2.set_ylabel("Broj stanica")
        ax2.axvline(6, color="gray", linestyle="--", linewidth=1)
    st.pyplot(fig2, width="stretch")

if show_side_colors and stats.n_cells > 0:
    st.markdown("**Mapa stanica obojena po broju stranica** (zeleno = heksagon / 6 stranica)")
    side_color_map = {3: (200, 50, 50), 4: (230, 140, 40), 5: (230, 210, 40),
                       6: (60, 180, 75), 7: (60, 140, 220), 8: (140, 60, 200)}
    colored = np.zeros((*stats.labels_img.shape, 3), dtype=np.uint8)
    ids = np.unique(stats.labels_img)
    from analysis import compute_neighbor_sides
    all_sides = compute_neighbor_sides(stats.labels_img, ids[ids > 0])
    for lbl, n_sides in all_sides.items():
        color = side_color_map.get(n_sides, (140, 140, 140))
        colored[stats.labels_img == lbl] = color
    colored[stats.boundaries] = [0, 0, 0]
    st.image(colored, width="stretch")

st.divider()
st.subheader("⬇️ Izvoz rezultata")

if stats.n_cells > 0:
    df = pd.DataFrame({
        "cell_id": np.arange(1, stats.n_cells + 1),
        "area_um2": stats.areas_um2,
        "n_sides": stats.sides,
        "centroid_row_px": [c[0] for c in stats.centroids],
        "centroid_col_px": [c[1] for c in stats.centroids],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    summary = pd.DataFrame([{
        "n_cells": stats.n_cells,
        "ECD_cells_per_mm2": stats.ecd,
        "mean_area_um2": stats.mean_area,
        "sd_area_um2": stats.sd_area,
        "min_area_um2": stats.min_area,
        "max_area_um2": stats.max_area,
        "CV_percent": stats.cv_percent,
        "HEX_percent": stats.hex_percent,
        "um_per_px": um_per_px,
        "calibration_profile": chosen_profile.name if chosen_profile else "(ručni unos)",
        "crop_left_px": crop_left,
        "crop_top_px": crop_top,
        "crop_right_px": crop_right,
        "crop_bottom_px": crop_bottom,
        "analyzed_area_mm2": stats.analyzed_area_mm2,
    }])
    summary_csv = summary.to_csv(index=False).encode("utf-8")

    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button("Preuzmi CSV po stanicama", csv_bytes, "cell_data.csv", "text/csv")
    dl2.download_button("Preuzmi sažetak (CSV)", summary_csv, "summary.csv", "text/csv")

    overlay_png = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_png.save(buf, format="PNG")
    dl3.download_button("Preuzmi sliku s granicama (PNG)", buf.getvalue(), "overlay.png", "image/png")
else:
    st.info("Nema detektiranih stanica za izvoz - prilagodite parametre segmentacije.")
