# ================================================================
# CLIMATEBERT GBR-WIDE SEASONAL EMBEDDINGS — FINAL VERSION
# Full semantic extraction:
#   - Prose text via pdfplumber
#   - Tables converted to natural language sentences
#   - Images/scanned pages via pytesseract OCR
#   - No keyword filtering — full semantic ClimateBERT
# ================================================================

# ── Cell 1: Install ─────────────────────────────────────────────
# !apt-get install -y tesseract-ocr -q
# !pip install pdfplumber transformers torch pandas tqdm pytesseract pdf2image pillow -q

# ── Cell 2: Imports ─────────────────────────────────────────────
import os, re, gc
import pdfplumber
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# OCR imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# ── Cell 3: Mount Drive ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ================================================================
# CONFIGURATION
# ================================================================

BASE_PATH = "/content/drive/MyDrive/coral_reports"

PDF_FOLDERS = {
    "AIMS":        BASE_PATH + "/AIMS Reports GBR",
    "ReefUpdates": BASE_PATH + "/Reef Updates Reports",
    "MMP":         BASE_PATH + "/MMP_Water Quality",
}

OUTPUT_PATH = BASE_PATH + "/coral_climatebert_gbr_embeddings.csv"

MAX_PAGES = {
    "AIMS":        9999,
    "ReefUpdates": 9999,
    "MMP":         80,    # skip raw data appendices
}

# Minimum text per page before OCR fallback kicks in
OCR_FALLBACK_THRESHOLD = 50   # chars — if pdfplumber gets less than this, try OCR

# ================================================================
# LOAD CLIMATEBERT DIRECTLY
# ================================================================

MODEL_NAME = "climatebert/distilroberta-base-climate-f"

print("⏳ Loading ClimateBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
cb_model  = AutoModel.from_pretrained(MODEL_NAME)
cb_model.eval()

device   = "cuda" if torch.cuda.is_available() else "cpu"
cb_model = cb_model.to(device)
print(f"✓ ClimateBERT loaded on {device}")

EMB_DIM = 768

# ================================================================
# EMBEDDING FUNCTION
# Mean pool last hidden state → L2 normalize
# ================================================================

def get_embedding(text):
    if not text or len(text.strip()) < 10:
        return np.zeros(EMB_DIM)

    inputs = tokenizer(
        text,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 512,
        padding        = True
    ).to(device)

    with torch.no_grad():
        outputs = cb_model(**inputs)

    last_hidden = outputs.last_hidden_state
    mask        = inputs["attention_mask"].unsqueeze(-1).float()
    pooled      = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    emb         = pooled.squeeze().cpu().numpy()

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb

# ================================================================
# TABLE → NATURAL LANGUAGE
# Converts table rows into readable sentences for ClimateBERT
# e.g. "DHW: 8.2. Bleaching: 45%. Status: Severe"
# ================================================================

def table_to_text(table):
    if not table or len(table) < 2:
        return ""

    # First row = headers
    headers = [str(h).strip() if h else "" for h in table[0]]
    sentences = []

    for row in table[1:]:
        if not row:
            continue
        cells = [str(c).strip() if c else "" for c in row]

        # Skip rows that are all empty
        if not any(cells):
            continue

        parts = []
        for h, v in zip(headers, cells):
            if v and v.lower() not in ["none", "n/a", "-", ""]:
                if h:
                    parts.append(f"{h}: {v}")
                else:
                    parts.append(v)

        if parts:
            sentences.append(". ".join(parts))

    return " ".join(sentences)

# ================================================================
# OCR A SINGLE PAGE IMAGE
# Used when pdfplumber extracts too little text (image-based page)
# ================================================================

def ocr_page_image(img):
    try:
        text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()
    except Exception as e:
        return ""

# ================================================================
# PDF READER — prose + tables + OCR fallback
# ================================================================

def read_pdf(path, max_pages=9999):
    filename   = os.path.basename(path)
    all_chunks = []
    ocr_pages  = []   # pages where pdfplumber got too little text

    # ── Pass 1: pdfplumber for text + tables ────────────────────
    try:
        with pdfplumber.open(path) as pdf:
            n_pages = min(len(pdf.pages), max_pages)

            for i in range(n_pages):
                page = pdf.pages[i]
                page_chunks = []

                # Prose text
                txt = page.extract_text()
                if txt and len(txt.strip()) > OCR_FALLBACK_THRESHOLD:
                    page_chunks.append(txt.strip())
                else:
                    # Too little text — flag for OCR
                    ocr_pages.append(i)

                # Tables → natural language
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        table_txt = table_to_text(table)
                        if table_txt:
                            page_chunks.append(table_txt)
                except:
                    pass

                all_chunks.extend(page_chunks)

    except Exception as e:
        print(f"    ⚠ pdfplumber error: {e}")
        return ""

    # ── Pass 2: OCR for image-based pages ───────────────────────
    if ocr_pages:
        print(f"    OCR: {len(ocr_pages)} image-based pages detected, running OCR...")
        try:
            # Convert only the flagged pages to images
            # page numbers are 1-indexed for pdf2image
            page_nums = [p+1 for p in ocr_pages if p < max_pages]

            images = convert_from_path(
                path,
                first_page = min(page_nums),
                last_page  = max(page_nums),
                dpi        = 200,           # balance speed vs accuracy
            )

            for img in images:
                ocr_text = ocr_page_image(img)
                if ocr_text and len(ocr_text) > 20:
                    all_chunks.append(ocr_text)
                del img
            gc.collect()

        except Exception as e:
            print(f"    ⚠ OCR failed: {e}")

    return "\n".join(all_chunks)

# ================================================================
# SMART TRUNCATION
# Start + middle + end for representative coverage
# 3500 chars ≈ 500 tokens — fits ClimateBERT's 512 token limit
# ================================================================

def smart_truncate(text, max_chars=3500):
    if len(text) <= max_chars:
        return text
    chunk = max_chars // 3
    mid_s = len(text) // 2 - chunk // 2
    return text[:chunk] + " " + text[mid_s:mid_s+chunk] + " " + text[-chunk:]

# ================================================================
# SEASON HELPER
# ================================================================

MONTHS = {
    "january":"01","february":"02","march":"03","april":"04",
    "may":"05","june":"06","july":"07","august":"08",
    "september":"09","october":"10","november":"11","december":"12"
}
VALID_YEARS = [str(y) for y in range(2018, 2026)]

def get_season(date_str):
    try:
        year  = int(date_str[:4])
        month = int(date_str[5:7])
    except:
        return None, None, None
    if month == 12:
        return "Summer", year+1, f"{year+1}-Summer"
    elif month in [1, 2]:
        return "Summer", year,   f"{year}-Summer"
    elif month in [3, 4, 5]:
        return "Autumn", year,   f"{year}-Autumn"
    elif month in [6, 7, 8]:
        return "Winter", year,   f"{year}-Winter"
    else:
        return "Spring", year,   f"{year}-Spring"

# ================================================================
# DATE EXTRACTOR
# ================================================================

def extract_date(text, filename, source_type=""):
    # Normalize em dash and en dash → regular hyphen
    # Fixes "2018–19" in MMP report titles being missed
    raw = (filename + " " + text[:6000]).lower()
    raw = raw.replace("–", "-").replace("—", "-")

    # ── MMP-specific: year range in filename e.g. 2018-19, 2019-2020 ──
    # MMP reports run July-June financial year
    # Take START year + July → maps to Winter season correctly
    if source_type == "MMP":
        fname = filename.lower().replace("–", "-").replace("—", "-")
        hit   = re.search(r'(20\d{2})[-_](20\d{2}|\d{2})', fname)
        if hit:
            return f"{hit.group(1)}-07"   # July of start year → Winter

    # ── Month name + year ───────────────────────────────────────────
    hit = re.search(
        r'(?:\d{1,2}\s+)?(january|february|march|april|may|june|'
        r'july|august|september|october|november|december)\s+(20\d{2})',
        raw
    )
    if hit:
        return f"{hit.group(2)}-{MONTHS[hit.group(1)]}"

    # ── Year range (em dash already normalized to hyphen) ───────────
    # Take START year to avoid wrong year assignment
    hit = re.search(r'(20\d{2})-(20\d{2}|\d{2})', raw)
    if hit:
        return f"{hit.group(1)}-07"   # start year + July

    # ── Plain year — check filename first, then body ─────────────────
    fname_lower = filename.lower()
    for y in reversed(VALID_YEARS):
        if y in fname_lower:
            return f"{y}-07"

    for y in reversed(VALID_YEARS):
        if y in raw:
            return f"{y}-06"

    return None

# ================================================================
# PROCESS ONE PDF
# ================================================================

def get_mmp_season_keys(filename):
    """
    MMP reports cover a full financial year (July → June).
    e.g. 2018-19 covers: 2018-Winter, 2018-Spring, 2019-Summer, 2019-Autumn
    Returns all season_keys the report covers.
    """
    fname = filename.lower().replace("–", "-").replace("—", "-")
    hit   = re.search(r'(20\d{2})[-_](20\d{2}|\d{2})', fname)
    if not hit:
        return None
    start = int(hit.group(1))
    end   = start + 1
    return [
        (f"{start}-07", f"{start}-Winter", "Winter", start),
        (f"{start}-09", f"{start}-Spring", "Spring", start),
        (f"{end}-01",   f"{end}-Summer",   "Summer", end),
        (f"{end}-03",   f"{end}-Autumn",   "Autumn", end),
    ]


def process_file(path, source_type):
    filename = os.path.basename(path)
    max_pg   = MAX_PAGES.get(source_type, 9999)

    print(f"\n  📄 {filename}")
    raw_text = read_pdf(path, max_pg)

    if len(raw_text.strip()) < 100:
        print(f"    ⚠ Too little text extracted, skipping")
        return []

    embed_text = smart_truncate(raw_text, max_chars=3500)
    print(f"    Extracted chars: {len(raw_text)}")

    # ── MMP: broadcast to all 4 seasons the report covers ───────
    if source_type == "MMP":
        season_keys = get_mmp_season_keys(filename)
        if season_keys:
            rows = []
            for date, season_key, season, year in season_keys:
                if 2018 <= year <= 2025:
                    print(f"    → {season_key}")
                    rows.append({
                        "date":        date,
                        "season_key":  season_key,
                        "season":      season,
                        "year":        year,
                        "source_type": source_type,
                        "source_file": filename,
                        "embed_text":  embed_text,
                        "raw_chars":   len(raw_text),
                    })
            return rows

    # ── AIMS + ReefUpdates: single season per report ─────────────
    date = extract_date(raw_text, filename, source_type)
    if not date:
        print(f"    ⚠ No date found, skipping")
        return []

    season, year, season_key = get_season(date)
    if not season_key:
        return []

    print(f"    Date: {date} → {season_key}")
    return [{
        "date":        date,
        "season_key":  season_key,
        "season":      season,
        "year":        year,
        "source_type": source_type,
        "source_file": filename,
        "embed_text":  embed_text,
        "raw_chars":   len(raw_text),
    }]

# ================================================================
# MAIN — PROCESS ALL PDFS
# ================================================================

all_rows = []

for source_type, folder in PDF_FOLDERS.items():
    if not os.path.exists(folder):
        print(f"\n⚠ Folder not found: {folder}")
        continue

    pdfs = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ])

    print(f"\n📁 {source_type} — {len(pdfs)} PDFs")

    for pdf_path in tqdm(pdfs, desc=source_type):
        try:
            rows = process_file(pdf_path, source_type)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ✗ Error on {os.path.basename(pdf_path)}: {e}")
        gc.collect()

# ================================================================
# BUILD + AGGREGATE DATAFRAME
# ================================================================

df = pd.DataFrame(all_rows)

if len(df) == 0:
    print("\n❌ No rows extracted. Check PDF_FOLDERS paths.")
else:
    df = df[df["year"].between(2018, 2025)]
    df = df.drop_duplicates(subset=["season_key", "source_file"])

    print(f"\n✓ {len(df)} file-level rows extracted")
    print(df[["season_key", "source_type", "raw_chars"]].to_string())

    # Aggregate: join text from multiple PDFs covering same season
    seasonal = (
        df.groupby(["season_key", "season", "year"])
        .agg(
            embed_text   = ("embed_text",  lambda x: " ".join(x)),
            sources_used = ("source_type", lambda x: " | ".join(sorted(set(x)))),
            source_files = ("source_file", lambda x: " | ".join(x)),
        )
        .reset_index()
        .sort_values(["year", "season"])
        .reset_index(drop=True)
    )

    # Final truncation after combining sources
    seasonal["embed_text"] = seasonal["embed_text"].apply(
        lambda t: smart_truncate(t, max_chars=3500)
    )

    print(f"\n✓ {len(seasonal)} seasonal rows (GBR-wide)")
    print(seasonal[["season_key", "sources_used"]].to_string())

    # ================================================================
    # EMBED EACH SEASON WITH CLIMATEBERT
    # ================================================================

    print(f"\n⏳ Embedding {len(seasonal)} seasonal texts with ClimateBERT...")
    embeddings = []

    for i, row in tqdm(seasonal.iterrows(), total=len(seasonal), desc="Embedding"):
        emb = get_embedding(row["embed_text"])
        embeddings.append(emb)
        gc.collect()

    embeddings = np.array(embeddings)
    print(f"✓ Embedding shape: {embeddings.shape}")

    # ================================================================
    # SANITY CHECKS
    # ================================================================

    print("\n── Sanity Check 1: Norms (should all be ≈ 1.0) ──")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"All ≈ 1.0: {np.allclose(norms, 1.0, atol=0.01)}")
    print(f"Min: {norms.min():.4f}  Max: {norms.max():.4f}")

    print("\n── Sanity Check 2: Consecutive season similarity ──")
    for i in range(len(embeddings)-1):
        sim  = cos_sim([embeddings[i]], [embeddings[i+1]])[0][0]
        s1   = seasonal["season_key"].iloc[i]
        s2   = seasonal["season_key"].iloc[i+1]
        flag = " ⚠ suspiciously low" if sim < 0.4 else ""
        print(f"  {s1} → {s2}: {sim:.4f}{flag}")

    print("\n── Sanity Check 3: Most/least similar season pairs ──")
    sim_matrix = cos_sim(embeddings)
    keys  = seasonal["season_key"].tolist()
    pairs = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            pairs.append((sim_matrix[i,j], keys[i], keys[j]))
    pairs.sort()

    print("Most DIFFERENT seasons (lowest similarity):")
    for sim, s1, s2 in pairs[:3]:
        print(f"  {s1} vs {s2}: {sim:.4f}")
    print("Most SIMILAR seasons (highest similarity):")
    for sim, s1, s2 in pairs[-3:]:
        print(f"  {s1} vs {s2}: {sim:.4f}")

    # ================================================================
    # SAVE OUTPUT CSV
    # ================================================================

    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i+1}" for i in range(EMB_DIM)]
    )

    meta_cols = ["season_key", "season", "year", "sources_used", "source_files"]
    final = pd.concat(
        [seasonal[meta_cols].reset_index(drop=True), emb_df],
        axis=1
    )

    final.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Saved → {OUTPUT_PATH}")
    print(f"   Shape: {final.shape}")
    print(f"   Columns: season_key, season, year, sources_used, emb_1...emb_{EMB_DIM}")
    print(f"\n   Fusion merge key: season_key")
    print(f"   Both North and Central CNN rows get the same NLP vector per season")
    print("\nDone 🎉")
