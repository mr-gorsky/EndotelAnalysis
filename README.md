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
2. Slika se predobrađuje: grayscale, izglađivanje, izravnavanje neujednačene
   rasvjete slit-lampe, CLAHE kontrast.
3. Stanice se segmentiraju watershed algoritmom (granice = svijetli
   "zidovi" između stanica na specular fotografiji).
4. Izračunavaju se standardni morfometrijski parametri:
   - **ECD** – gustoća stanica (stanica/mm²), uz ručnu kalibraciju µm/piksel
   - **CV%** – koeficijent varijacije površine stanica (polimegatizam)
   - **HEX%** – postotak šesterokutnih stanica (heksagonalnost/pleomorfizam)
   - prosječna/min/max površina stanice
5. Prikazuje se preklop granica na originalnoj slici, histogrami, i mapa
   stanica obojena po broju stranica.
6. Rezultati (po stanici i sažetak) izvoze se kao CSV, a preklopna slika
   kao PNG.

## Ograničenja (važno)

- Ovo je **istraživački/edukativni prototip**, ne klinički validiran
  medicinski uređaj.
- Kalibracija (µm/piksel) mora se ručno unijeti za vaš sustav (slit-lampa +
  zum + kamera) da bi ECD i površine bile u točnim jedinicama.
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

- `app.py` – Streamlit sučelje
- `analysis.py` – jezgra obrade slike i morfometrije (može se testirati i
  bez Streamlita)
- `requirements.txt` – Python ovisnosti
