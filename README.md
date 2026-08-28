# Analiza korneal. endotela (Streamlit)

Prototip Streamlit aplikacije za analizu slika endotela rožnice snimljenih
specular tehnikom na slit-lampi (npr. telefonom preko oka slit-lampe).
Inspirirana pristupom iz rada *"Low-Cost, Smartphone-Based Specular Imaging
and Automated Analysis of the Corneal Endothelium"* (Translational Vision
Science & Technology, 2021; [PMC8024782](https://pmc.ncbi.nlm.nih.gov/articles/PMC8024782/)),
ali koristi vlastiti, otvoreno dokumentiran klasični image-processing pristup
(marker-controlled watershed), a ne njihov točan (patentirani) algoritam.

## Pokretanje

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikacija će se otvoriti u pregledniku na `http://localhost:8501`.

## Što radi

1. Učitate sliku endotela (PNG/JPG/TIFF/BMP) ili isprobate ugrađenu
   sintetsku demo sliku.
2. **Kalibracija (µm/piksel):**
   - Ne postoji javno objavljena specifikacija (rezolucija senzora / vidno
     polje po povećanju) za CSO-ovu kameru na biomikroskopu, pa aplikacija
     tu vrijednost ne pretpostavlja — nudi dva načina da je postaviš:
     - **Kalibracijski profili** (sidebar): unesi µm/px ili FOV (mm) ručno
       ako ga već znaš, po nazivu povećanja (npr. "40x"). Profil se
       prepoznaje automatski kad učitaš sliku iste rezolucije, i može se
       izvesti/uvesti kao JSON da ga sačuvaš između sesija.
     - **🔧 Kalibracija ravnalom**: snimi mjernu pločicu/mm papir kroz isti
       sustav pri istom povećanju (isti princip kao kalibracija preko mm
       papira u Phoenixu). Dva načina mjerenja:
       - **Linija (2 točke)** — povuci liniju preko poznate udaljenosti,
         upiši tu udaljenost u mm.
       - **Pravokutnik / mreža (4 kuta)** — klikneš redom 4 kuta poznatog
         pravokutnika (npr. kut kvadratića na mm papiru); precizniji je jer
         koristi sve 4 stranice odjednom i usput provjerava slažu li se
         X i Y os (upozorenje ako je razlika > 5%, znak da je neki kut
         krivo postavljen).
       Oba načina mogu odmah spremiti rezultat kao profil.
   - Ako slika ima drugačiju rezoluciju od profila (npr. izvezena u manjoj
     rezoluciji), aplikacija nudi proporcionalno preračunavanje uz tvoju
     potvrdu da je i dalje riječ o cijelom kadru (ne prethodno rezanom).
3. **✂️ Crop**: povuci pravokutnik preko slike (ili unesi brojeve ručno) da
   odabereš dio za analizu — npr. da izbjegneš neoštre rubove ili artefakte.
   Kalibracija (µm/px) ostaje ista prije i poslije cropa, jer crop samo bira
   piksele, ne mijenja njihovu veličinu.
4. Slika se predobrađuje: grayscale, izglađivanje, izravnavanje neujednačene
   rasvjete slit-lampe, CLAHE kontrast.
5. Stanice se segmentiraju watershed algoritmom (granice = svijetli
   "zidovi" između stanica na specular fotografiji).
6. Izračunavaju se standardni morfometrijski parametri:
   - **ECD** – gustoća stanica (stanica/mm²)
   - **CV%** – koeficijent varijacije površine stanica (polimegatizam)
   - **HEX%** – postotak šesterokutnih stanica (heksagonalnost/pleomorfizam)
   - prosječna/min/max površina stanice
7. Prikazuje se preklop granica na originalnoj slici, histogrami, i mapa
   stanica obojena po broju stranica.
8. Rezultati (po stanici i sažetak, uključujući koordinate cropa i naziv
   korištenog kalibracijskog profila) izvoze se kao CSV, a preklopna slika
   kao PNG.

## Ograničenja (važno)

- Ovo je **istraživački/edukativni prototip**, ne klinički validiran
  medicinski uređaj.
- CSO ne objavljuje javno specifikacije rezolucije/vidnog polja kamere na
  biomikroskopu — kalibraciju treba postaviti sam (već poznata vrijednost,
  ili alat za kalibraciju ravnalom), inače ECD/površine nisu u točnim
  jedinicama (CV% i HEX% ostaju smisleni jer ne ovise o apsolutnoj skali).
- Kalibracijski profili spremaju se samo za trajanja sesije preglednika —
  izvezi ih kao JSON da ih zadržiš za sljedeći put.
- Kvaliteta segmentacije ovisi o kontrastu/oštrini/rasvjeti slike – za
  slabije slike prilagodite parametre u lijevom izborniku (kontrast,
  izglađivanje, pomak praga, min. veličina stanice).
- Broj "stranica" stanice računa se preko susjednih labela u mozaiku, što je
  praktična aproksimacija stvarne "triple-point" metode iz literature.
- Za pouzdanu morfometriju obično je potrebno ≥ 75–100 analiziranih stanica.

## Deploy na Streamlit Community Cloud

1. Kreiraj prazan repozitorij na GitHubu (npr. `endothelium-analyzer`),
   javan (za besplatni tier Community Cloud treba javni repo, osim ako
   imaš plaćeni/verificirani račun koji dopušta privatne).
2. U ovoj mapi (već je `git init`-ana i sve je commit-ano):
   ```bash
   git remote add origin https://github.com/<tvoj-username>/<naziv-repoa>.git
   git branch -M main
   git push -u origin main
   ```
3. Idi na [share.streamlit.io](https://share.streamlit.io), prijavi se
   GitHub računom.
4. "New app" → odaberi repo i `main` granu → main file path: `app.py` →
   Deploy.
5. Nakon par minuta dobiješ javni URL oblika
   `https://<naziv>.streamlit.app`.

**Napomena o privatnosti:** ako planiraš testirati sa stvarnim slikama
pacijenata, imaj na umu da besplatni Streamlit Community Cloud pokreće
app na dijeljenoj javnoj infrastrukturi i URL je javno dostupan svima s
linkom (osim ako postaviš pristupna ograničenja). Za stvarne kliničke
podatke razmisli o privatnom deployu (npr. lokalno, interni server, ili
Streamlit-ov plaćeni private app tier) umjesto javnog demo linka.

## Struktura projekta

- `app.py` – Streamlit sučelje (upload, kalibracija, crop, prikaz, izvoz)
- `analysis.py` – jezgra obrade slike i morfometrije (može se testirati i
  bez Streamlita)
- `calibration.py` – kalibracijski profili i računanje µm/px (JSON
  import/export, kalibracija preko dvije točke poznate udaljenosti)
- `requirements.txt` – Python ovisnosti
