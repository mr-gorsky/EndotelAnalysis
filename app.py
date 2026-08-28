import io
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from analysis import PipelineParams, load_grayscale, preprocess, analyze

st.set_page_config(page_title="Analiza korneal. endotela", layout="wide")

st.title("🔬 Analiza slike korneal. endotela (slit-lamp / specular)")

with st.expander("ℹ️ O aplikaciji i ograničenjima", expanded=False):
    st.markdown(
        """
Ova aplikacija je **istraživački / edukativni prototip**, inspiriran pristupom
opisanim u radu *"Low-Cost, Smartphone-Based Specular Imaging and Automated
Analysis of the Corneal Endothelium"* (Translational Vision Science &
Technology, 2021,
[PMC8024782](https://pmc.ncbi.nlm.nih.gov/articles/PMC8024782/)) — smartphone
slit-lamp specular fotografija + automatska segmentacija stanica endotela.

**Nije riječ o rekonstrukciji njihovog točnog (patentiranog) algoritma.**
Koristi se standardna, otvoreno dokumentirana klasična tehnika segmentacije
("marker-controlled watershed" na distance-transformu), a broj "stranica"
stanice (za heksagonalnost) procjenjuje se preko broja susjednih stanica u
mozaiku — što je uobičajena zamjena za "triple-point" metodu iz literature.

**Ograničenja koja treba imati na umu:**
- Rezultati ovise o kvaliteti/kontrastu slike i o parametrima koje namjestite.
- Kalibracija (µm/piksel) mora biti unesena ručno — bez nje su apsolutne
  vrijednosti (ECD, površine) neupotrebljive, iako je CV% i HEX% i dalje
  informativan.
- Ovo **nije klinički validiran medicinski uređaj** i ne smije se koristiti
  kao zamjena za specular mikroskopiju u kliničkom odlučivanju bez
  neovisne validacije na poznatim uzorcima.
        """
    )

# ---------------------------------------------------------------- sidebar --
st.sidebar.header("1. Učitaj sliku")
uploaded = st.sidebar.file_uploader(
    "Slika endotela (specular / slit-lamp)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
)

use_demo = False
if uploaded is None:
    use_demo = st.sidebar.checkbox("Koristi sintetsku demo sliku", value=True)

st.sidebar.header("2. Kalibracija")
um_per_px = st.sidebar.number_input(
    "Mikrometara po pikselu (µm/px)",
    min_value=0.01, max_value=50.0, value=1.0, step=0.01,
    help="Ako ne znate točnu kalibraciju za vaš slit-lamp + zum + kameru, "
         "ECD i površine neće biti u točnim jedinicama — ali CV% i HEX% "
         "ostaju smisleni jer ne ovise o apsolutnoj skali."
)

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
gray = None
if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_arr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img_arr is None:
        st.error("Nije moguće učitati sliku. Provjerite format datoteke.")
    else:
        if img_arr.ndim == 3:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        gray = load_grayscale(img_arr)
elif use_demo:
    gray = make_demo_image()
    st.info("Prikazana je **sintetska demo slika** (nije stvarni pacijent) "
            "samo da vidite kako alat radi. Učitajte pravu sliku endotela "
            "u lijevom izborniku za stvarnu analizu.")

if gray is None:
    st.warning("Učitajte sliku endotela u lijevom izborniku da biste započeli analizu.")
    st.stop()

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
col1.image(gray, caption="Originalna (grayscale)", width="stretch", clamp=True)
col2.image(enhanced, caption="Nakon predobrade (CLAHE + normalizacija)", width="stretch", clamp=True)

overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
overlay[stats.boundaries] = [255, 60, 60]
col3.image(overlay, caption="Segmentirane granice stanica", width="stretch")

st.subheader("📊 Morfometrijski parametri")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Broj analiziranih stanica", f"{stats.n_cells}")
m2.metric("ECD (stanica/mm²)", f"{stats.ecd:,.0f}" if um_per_px != 1.0 or uploaded else f"{stats.ecd:,.0f} *")
m3.metric("Prosj. površina (µm²)", f"{stats.mean_area:,.1f}")
m4.metric("CV % (polimegatizam)", f"{stats.cv_percent:,.1f}%")
m5.metric("HEX % (heksagonalnost)", f"{stats.hex_percent:,.1f}%")

if um_per_px == 1.0:
    st.caption("*ECD i površine koriste zadanu kalibraciju 1 µm/px — unesite "
               "stvarnu kalibraciju vašeg sustava za točne apsolutne vrijednosti.")

if stats.n_cells < 50:
    st.warning(
        f"Analizirano je samo **{stats.n_cells}** stanica. Pouzdana klinička "
        "morfometrija endotela obično zahtijeva ≥ 75–100 stanica — smatrajte "
        "ove brojke orijentacijskima i pokušajte poboljšati kontrast/kvalitetu "
        "slike ili prilagoditi parametre segmentacije u izborniku."
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
    sides_by_label = {}
    ids = np.unique(stats.labels_img)
    # rebuild mapping from analyze() outputs (order-aligned with props of interior labels)
    # simplest robust approach: recompute quickly here for visualization only
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
