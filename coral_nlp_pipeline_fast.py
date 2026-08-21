# ============================================================
# CORALWATCH NLP PIPELINE - SEASONAL VERSION (FAST)
# Extracts: bleaching, species, stressors, severity
# Aggregates by: season + year + region (for fusion alignment)
# Output: seasonal feature scores + MiniLM embeddings
#         + all matched sentences saved for inspection
# ============================================================

# !pip install pdfplumber sentence-transformers pandas tqdm openpyxl

import os, re, gc
import pdfplumber
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION - change these paths
# ============================================================

BASE_PATH   = "/content/drive/MyDrive/coral reports"

PDF_FOLDERS = [
    BASE_PATH + "/AIMS Reports GBR",
    BASE_PATH + "/Reef Updates Reports",
    BASE_PATH + "/MMP_Water Quality",
]

OUTPUT_CSV        = BASE_PATH + "/coral_nlp_seasonal_features.csv"
OUTPUT_XLSX       = BASE_PATH + "/coral_nlp_seasonal_features.xlsx"
OUTPUT_SENTENCES  = BASE_PATH + "/coral_extracted_sentences.csv"
OUTPUT_EMBEDDINGS = BASE_PATH + "/coral_minilm_seasonal_embeddings.csv"

# ============================================================
# MONTH MAP
# ============================================================

MONTHS = {
    "january":"01","february":"02","march":"03","april":"04",
    "may":"05","june":"06","july":"07","august":"08",
    "september":"09","october":"10","november":"11","december":"12"
}

VALID_YEARS = [str(y) for y in range(2018, 2026)]

# ============================================================
# SEASON ASSIGNMENT
# GBR-specific seasons:
#   Summer  Dec-Feb  peak thermal stress / active bleaching
#   Autumn  Mar-May  post-bleaching / recovery observations
#   Winter  Jun-Aug  low stress baseline
#   Spring  Sep-Nov  pre-stress buildup
# Note: Dec belongs to NEXT year's summer
#       e.g. Dec 2021 -> Summer 2022
# ============================================================

def get_season(date_str):
    try:
        year  = int(date_str[:4])
        month = int(date_str[5:7])
    except:
        return None, None, None
    if month in [12, 1, 2]:
        season = "Summer"
        season_year = year + 1 if month == 12 else year
    elif month in [3, 4, 5]:
        season = "Autumn"
        season_year = year
    elif month in [6, 7, 8]:
        season = "Winter"
        season_year = year
    else:
        season = "Spring"
        season_year = year
    season_key = f"{season_year}-{season}"
    return season, season_year, season_key

# ============================================================
# REGION TERMS - specific multi-word phrases only
# ============================================================

REGION_TERMS = {
    "North": [
        "northern gbr", "northern great barrier reef",
        "far northern", "far north",
        "lizard island", "cooktown", "cape york",
        "port douglas", "cairns region",
        "north region", "north gbr",
    ],
    "Central": [
        "central gbr", "central great barrier reef",
        "central region",
        "townsville", "burdekin",
        "whitsunday", "mackay",
        "central section",
    ],
}

# ============================================================
# SPECIES DICTIONARY
# High = thermally sensitive (bleach fast)
# Moderate = intermediate
# Low = thermally tolerant (bleach slowly)
# ============================================================

SPECIES = {
    "high_sensitivity": [
        "acropora", "staghorn coral", "table coral",
        "pocillopora", "seriatopora", "stylophora",
        "branching coral", "plate coral",
    ],
    "moderate_sensitivity": [
        "montipora", "turbinaria", "fungia",
        "soft coral", "alcyonarian",
    ],
    "low_sensitivity": [
        "porites", "massive coral", "faviidae",
        "lobophyllia", "platygyra", "diploastrea",
        "encrusting coral",
    ],
}

# ============================================================
# STRESSOR KEYWORDS
# ============================================================

KEYWORDS = {
    "bleaching": [
        "bleaching", "coral bleaching", "bleached coral",
        "fully bleached", "partially bleached", "paling",
        "whitening", "bleach event",
    ],
    "heat_stress": [
        "heat stress", "thermal stress", "marine heatwave",
        "degree heating weeks", "dhw", "elevated temperature",
        "sea surface temperature anomaly", "ssta", "warming",
    ],
    "turbidity": [
        "turbidity", "murky", "cloudy water",
        "poor visibility", "reduced visibility",
        "low visibility", "poor water clarity",
    ],
    "sediment": [
        "sediment", "fine sediment", "suspended solids",
        "runoff", "terrestrial runoff", "sedimentation",
    ],
    "flood": [
        "flood", "flood plume", "flooding",
        "freshwater intrusion", "river discharge",
        "river plume", "freshwater",
    ],
    "cots": [
        "cots", "crown of thorns", "crown-of-thorns",
        "acanthaster", "starfish outbreak",
    ],
    "cyclone": [
        "cyclone", "tropical cyclone", "storm damage",
        "jasper", "koji", "alfred", "wind damage",
    ],
    "disease": [
        "disease", "coral disease", "tissue loss",
        "white syndrome", "brown band", "growth anomaly",
    ],
}

# ============================================================
# SEVERITY WORDS
# ============================================================

SEVERITY_MAP = {
    "no":          0.00,
    "not":         0.00,
    "without":     0.00,
    "none":        0.00,
    "little":      0.10,
    "minor":       0.33,
    "low":         0.33,
    "limited":     0.33,
    "some":        0.50,
    "moderate":    0.66,
    "medium":      0.66,
    "significant": 0.75,
    "high":        1.00,
    "severe":      1.00,
    "extreme":     1.00,
    "mass":        1.00,
    "widespread":  1.00,
    "extensive":   1.00,
}

NEGATIONS = ["no ", "not ", "without ", "none ", "absence of ", "no evidence of "]

# ============================================================
# DATE EXTRACTOR
# ============================================================

def extract_date(text, filename):
    raw = (filename + " " + text[:6000]).lower()

    # Format: 03 April 2025
    hit = re.search(
        r'\d{1,2}\s+'
        r'(january|february|march|april|may|june|'
        r'july|august|september|october|november|december)'
        r'\s+(20\d{2})',
        raw
    )
    if hit:
        return f"{hit.group(2)}-{MONTHS[hit.group(1)]}"

    # Format: April 2025
    hit = re.search(
        r'(january|february|march|april|may|june|'
        r'july|august|september|october|november|december)'
        r'\s+(20\d{2})',
        raw
    )
    if hit:
        return f"{hit.group(2)}-{MONTHS[hit.group(1)]}"

    # Format: year range 2024-25
    hit = re.search(r'(20\d{2})[-_/](20\d{2}|\d{2})', raw)
    if hit:
        y2 = hit.group(2)
        if len(y2) == 2:
            y2 = hit.group(1)[:2] + y2
        return f"{y2}-01"

    # Fallback: plain year
    for y in reversed(VALID_YEARS):
        if y in raw:
            return f"{y}-01"

    return None

# ============================================================
# PDF READER - FAST VERSION with smart page limits
# MMP/Water Quality reports: first 80 pages only
#   (summaries + results are in first ~80 pages;
#    remaining 200+ pages are raw data tables / appendices)
# AIMS / Reef Updates: all pages (shorter docs, need full text)
# ============================================================

def read_pdf(path):
    chunks = []
    path_str = str(path).lower()

    if "mmp" in path_str or "water quality" in path_str or "monitoring" in path_str:
        max_pages = 80
    else:
        max_pages = 9999  # all pages

    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                txt = page.extract_text()
                if txt:
                    chunks.append(txt)
    except Exception as e:
        print(f"  Skipped {path}: {e}")
        return ""
    return "\n".join(chunks)

# ============================================================
# SENTENCE SPLITTER
# ============================================================

def split_sentences(text):
    raw = re.split(r'(?<=[.!?])\s+|\n', text)
    sentences = []
    for s in raw:
        s = s.strip()
        s = re.sub(r'\s+', ' ', s)
        if len(s) > 15:
            sentences.append(s)
    return sentences

# ============================================================
# REGION FILTER - sentence level, strict matching
# ============================================================

def filter_region_sentences(sentences, region):
    keys = REGION_TERMS[region]
    matched = []
    for s in sentences:
        s_low = s.lower()
        if any(k in s_low for k in keys):
            matched.append(s)
    return matched

# ============================================================
# NEGATION CHECK
# ============================================================

def is_negated(sentence, keyword):
    s = sentence.lower()
    idx = s.find(keyword)
    if idx == -1:
        return False
    context = s[max(0, idx-40):idx]
    return any(neg in context for neg in NEGATIONS)

# ============================================================
# SCORE A VARIABLE FROM SENTENCES
# ============================================================

def score_variable(sentences, keywords):
    best_score = 0.0
    matched_sentences = []

    for s in sentences:
        s_low = s.lower()
        has_keyword = any(kw in s_low for kw in keywords)
        if not has_keyword:
            continue
        negated = any(is_negated(s, kw) for kw in keywords if kw in s_low)
        if negated:
            continue
        matched_sentences.append(s)
        score = 0.33
        for sev_word, sev_score in SEVERITY_MAP.items():
            if sev_word in s_low:
                score = max(score, sev_score)
        best_score = max(best_score, score)

    return best_score, matched_sentences

# ============================================================
# SPECIES SCORER
# ============================================================

def score_species(sentences):
    vulnerability = 0.0
    species_bleaching = 0.0
    matched = []

    for s in sentences:
        s_low = s.lower()
        found_species = None
        sp_score = 0.0

        for sp in SPECIES["high_sensitivity"]:
            if sp in s_low:
                found_species = sp
                sp_score = 1.0
                break
        if not found_species:
            for sp in SPECIES["moderate_sensitivity"]:
                if sp in s_low:
                    found_species = sp
                    sp_score = 0.66
                    break
        if not found_species:
            for sp in SPECIES["low_sensitivity"]:
                if sp in s_low:
                    found_species = sp
                    sp_score = 0.33
                    break

        if found_species:
            vulnerability = max(vulnerability, sp_score)
            matched.append(s)
            if any(bw in s_low for bw in ["bleach", "pale", "white", "stress"]):
                if not is_negated(s, found_species):
                    species_bleaching = max(species_bleaching, sp_score)

    return vulnerability, species_bleaching, matched

# ============================================================
# PROCESS ONE PDF FILE
# ============================================================

def process_file(path):
    filename = os.path.basename(path)
    print(f"  Reading: {filename}")

    raw_text = read_pdf(path)

    if len(raw_text.strip()) < 100:
        return [], []

    date = extract_date(raw_text, filename)
    if date is None:
        print(f"    No date found, skipping.")
        return [], []

    sentences = split_sentences(raw_text)
    print(f"    Sentences: {len(sentences)}, Date: {date}")

    feature_rows = []
    sentence_rows = []

    for region in ["North", "Central"]:
        region_sents = filter_region_sentences(sentences, region)
        if len(region_sents) < 5:
            region_sents = sentences

        row = {
            "date": date,
            "source_file": filename,
            "region": region,
            "sentence_count": len(region_sents),
        }

        all_matched = []
        for var, kws in KEYWORDS.items():
            score, matched = score_variable(region_sents, kws)
            row[var] = score
            for s in matched:
                sentence_rows.append({
                    "date": date,
                    "source_file": filename,
                    "region": region,
                    "variable": var,
                    "sentence": s,
                })
            all_matched.extend(matched)

        vuln, sp_bleach, sp_matched = score_species(region_sents)
        row["species_vulnerability"] = vuln
        row["species_bleaching"] = sp_bleach

        for s in sp_matched:
            sentence_rows.append({
                "date": date,
                "source_file": filename,
                "region": region,
                "variable": "species",
                "sentence": s,
            })

        all_matched_text = " ".join(set(all_matched + sp_matched))
        row["matched_text"] = all_matched_text if all_matched_text else " ".join(region_sents[:30])

        feature_rows.append(row)

    return feature_rows, sentence_rows

# ============================================================
# MAIN
# ============================================================

all_feature_rows = []
all_sentence_rows = []

for folder in PDF_FOLDERS:
    if not os.path.exists(folder):
        print(f"Folder not found, skipping: {folder}")
        continue

    pdfs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]

    print(f"\nFolder: {folder} — {len(pdfs)} PDFs")

    for pdf_path in tqdm(pdfs):
        try:
            feat_rows, sent_rows = process_file(pdf_path)
            all_feature_rows.extend(feat_rows)
            all_sentence_rows.extend(sent_rows)
        except Exception as e:
            print(f"  Error: {e}")
        gc.collect()

# ============================================================
# BUILD FEATURE DATAFRAME
# ============================================================

df = pd.DataFrame(all_feature_rows)

if len(df) == 0:
    print("No rows extracted. Check folder paths.")
else:
    df = df.drop_duplicates(subset=["date", "source_file", "region"])
    df = df.dropna(subset=["date"])
    df = df[df["date"].str[:4].astype(int).between(2018, 2025)]
    df = df.sort_values(["date", "region"]).reset_index(drop=True)

    print(f"\nFeature rows: {df.shape}")
    print(df[["date","region","bleaching","species_vulnerability","species_bleaching","heat_stress","cots"]].head(20))

    # ============================================================
    # ADD SEASON LABELS
    # ============================================================

    df[["season","year","season_key"]] = df["date"].apply(
        lambda d: pd.Series(get_season(d))
    )
    df = df.dropna(subset=["season_key"])

    print("\nSeason distribution:")
    print(df.groupby(["season_key","region"]).size().reset_index(name="report_count"))

    # ============================================================
    # AGGREGATE TO SEASONAL
    # ============================================================

    numeric_cols = [
        "bleaching","heat_stress","turbidity","sediment",
        "flood","cots","cyclone","disease",
        "species_vulnerability","species_bleaching"
    ]

    text_agg = df.groupby(["season_key","region"])["matched_text"].apply(
        lambda x: " ".join(x)
    ).reset_index()

    score_agg = df.groupby(["season_key","region"])[numeric_cols].max().reset_index()
    season_meta = df[["season_key","season","year"]].drop_duplicates()

    seasonal = score_agg.merge(text_agg, on=["season_key","region"])
    seasonal = seasonal.merge(season_meta, on="season_key")
    seasonal = seasonal[
        ["season_key","season","year","region"] + numeric_cols + ["matched_text"]
    ].sort_values(["year","season","region"]).reset_index(drop=True)

    print(f"\nSeasonal aggregated rows: {seasonal.shape}")
    print(seasonal[["season_key","region","bleaching","species_vulnerability","species_bleaching","heat_stress"]].to_string())

    # ============================================================
    # SAVE FEATURE CSV + XLSX
    # ============================================================

    seasonal.drop(columns=["matched_text"]).to_csv(OUTPUT_CSV, index=False)
    seasonal.drop(columns=["matched_text"]).to_excel(OUTPUT_XLSX, index=False)
    print(f"\nSaved features: {OUTPUT_CSV}")

    # ============================================================
    # SAVE EXTRACTED SENTENCES
    # ============================================================

    sent_df = pd.DataFrame(all_sentence_rows)
    sent_df = sent_df.drop_duplicates()
    sent_df = sent_df.sort_values(["date","region","variable"]).reset_index(drop=True)
    sent_df.to_csv(OUTPUT_SENTENCES, index=False)
    print(f"Saved sentences: {OUTPUT_SENTENCES} ({len(sent_df)} rows)")

    # ============================================================
    # MINILM EMBEDDINGS
    # ============================================================

    print("\nGenerating MiniLM embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = seasonal["matched_text"].tolist()
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=True)
    print(f"Embedding shape: {embeddings.shape}")

    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i+1}" for i in range(embeddings.shape[1])]
    )

    final_emb = pd.concat(
        [seasonal[["season_key","season","year","region"] + numeric_cols], emb_df],
        axis=1
    )

    final_emb.to_csv(OUTPUT_EMBEDDINGS, index=False)
    print(f"Saved embeddings: {OUTPUT_EMBEDDINGS}")
    print(f"Final embedding shape: {final_emb.shape}")
    print("\nColumns for fusion merge: season_key + region")
    print("Done.")
