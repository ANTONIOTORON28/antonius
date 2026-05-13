# =========================================================
# SISTEM REKOMENDASI PAKET WISATA KAPAL PHINISI
# Implementasi Sentence-BERT + Content-Based Filtering
# Labuan Bajo — Full Card Visualization
# =========================================================
#
# INSTALL:
#   pip install streamlit pandas sentence-transformers scikit-learn pillow torch
#
# JALANKAN:
#   streamlit run app.py
#
# STRUKTUR CSV (dataset_kapal_preprocessing.csv):
#   nama_paket | nama_kapal | kategori | harga | durasi | kapasitas
#   destinasi  | fasilitas  | layanan  | deskripsi | image_url | link | content
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Phinisi · Sistem Rekomendasi Labuan Bajo",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_kapal_preprocessing.csv")
    df["content"] = df["content"].fillna("")
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def build_corpus_embeddings(_model, contents: list):
    return _model.encode(contents, show_progress_bar=False)

df         = load_data()
model      = load_model()
corpus_emb = build_corpus_embeddings(model, df["content"].tolist())

# ─────────────────────────────────────────
# FUNGSI REKOMENDASI
# ─────────────────────────────────────────
def recommend(query: str, top_k: int = 5, kategori: str = "Semua") -> pd.DataFrame:
    q_emb  = model.encode([query])
    scores = cosine_similarity(q_emb, corpus_emb)[0]
    out = df.copy()
    out["skor"] = scores
    if kategori != "Semua":
        out = out[out["kategori"] == kategori]
    return out.sort_values("skor", ascending=False).head(top_k).reset_index(drop=True)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def kategori_color(k):
    ku = str(k).upper()
    if   "VVIP"     in ku: return "#b45309", "#fef3c7", "👑"
    elif "VIP"      in ku: return "#6d28d9", "#ede9fe", "💎"
    elif "STANDARD" in ku or "STD" in ku: return "#065f46", "#d1fae5", "⚡"
    else:                  return "#0c4a6e", "#e0f2fe", "🚢"

def rank_badge(i):
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    return medals.get(i, f"#{i+1}")

def parse_chips(raw: str) -> list:
    """Safely parse comma-separated string into list."""
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "results"    not in st.session_state: st.session_state.results    = None
if "last_query" not in st.session_state: st.session_state.last_query = ""

# ─────────────────────────────────────────
# GLOBAL CSS  (tema TERANG / Light)
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* ── RESET / BASE ────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #f0f6ff;
    color: #1e3a5a;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── SCROLLBAR ────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #e8f0fb; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #0ea5e9, #0369a1);
    border-radius: 99px;
}

/* ── LAYOUT ───────────────────────────────────────── */
.block-container { padding: 0 2.5rem 6rem !important; max-width: 1340px !important; }
[data-testid="column"] { padding: 0 10px !important; }

/* ════════════════════════════════════════════════════
   HERO
════════════════════════════════════════════════════ */
.hero {
    text-align: center;
    padding: 52px 16px 0;
    margin-bottom: 38px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(14,165,233,.10);
    border: 1px solid rgba(14,165,233,.35);
    border-radius: 99px;
    padding: 7px 22px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #0369a1;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(28px, 4.5vw, 54px);
    font-weight: 700;
    line-height: 1.12;
    background: linear-gradient(140deg, #0c4a6e 0%, #0369a1 45%, #0ea5e9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
}
.hero-sub {
    font-size: 13px;
    color: #64748b;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-desc {
    font-size: 15px;
    font-weight: 300;
    color: #475569;
    line-height: 1.75;
    max-width: 540px;
    margin: 0 auto;
}
.hero-line {
    width: 90px;
    height: 2px;
    margin: 26px auto 0;
    background: linear-gradient(90deg, transparent, #0ea5e9 50%, transparent);
}

/* ════════════════════════════════════════════════════
   QUERY PANEL
════════════════════════════════════════════════════ */
.qpanel {
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 24px;
    padding: 28px 30px 22px;
    margin-bottom: 18px;
    box-shadow: 0 4px 24px rgba(14,165,233,.08), 0 1px 3px rgba(0,0,0,.04);
}
.qhead {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: #0369a1;
    margin-bottom: 4px;
}
.qhint {
    font-size: 13px;
    color: #64748b;
    font-weight: 300;
    margin-bottom: 18px;
    line-height: 1.6;
}
.example-label {
    font-size: 11px;
    color: #94a3b8;
    letter-spacing: .10em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 9px;
}
.example-chip {
    display: inline-block;
    padding: 5px 12px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    font-size: 12px;
    color: #0369a1;
    margin: 3px;
}

/* ── STREAMLIT TEXTAREA ───────────────────────────── */
.stTextArea textarea {
    background: #f8faff !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 16px !important;
    color: #1e3a5a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    padding: 16px 20px !important;
    transition: border-color .25s, box-shadow .25s;
}
.stTextArea textarea:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,.15) !important;
}
.stTextArea textarea::placeholder { color: #94a3b8 !important; }
.stTextArea label { display: none !important; }

/* ── SELECTBOX ─────────────────────────────────────── */
.stSelectbox > div > div {
    background: #f8faff !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 14px !important;
    color: #1e3a5a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stSelectbox label {
    color: #64748b !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: .14em !important;
    text-transform: uppercase !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── NUMBER INPUT ──────────────────────────────────── */
.stNumberInput > div > div {
    background: #f8faff !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 14px !important;
    color: #1e3a5a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stNumberInput label {
    color: #64748b !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: .14em !important;
    text-transform: uppercase !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── BUTTON ────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0369a1 0%, #0284c7 60%, #0ea5e9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 13px 32px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: .04em !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(14,165,233,.30) !important;
    transition: all .25s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 32px rgba(14,165,233,.48) !important;
    transform: translateY(-2px) !important;
}

/* ════════════════════════════════════════════════════
   RESULT HEADER
════════════════════════════════════════════════════ */
.res-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    margin: 36px 0 22px;
    padding-bottom: 14px;
    border-bottom: 2px solid #e0f2fe;
}
.res-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #0c4a6e;
}
.res-qtag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 6px 14px;
    font-size: 13px;
    color: #0369a1;
    font-weight: 500;
    max-width: 460px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ════════════════════════════════════════════════════
   VESSEL CARD
════════════════════════════════════════════════════ */
.vcard {
    background: #ffffff;
    border-radius: 24px;
    border: 1px solid #e0f2fe;
    overflow: hidden;
    margin-bottom: 24px;
    transition: transform .32s cubic-bezier(.22,.68,0,1.15), box-shadow .32s;
    box-shadow: 0 4px 20px rgba(14,165,233,.08), 0 1px 4px rgba(0,0,0,.04);
}
.vcard:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(14,165,233,.16), 0 2px 8px rgba(0,0,0,.06);
}
.vcard-top-bar {
    height: 4px;
    background: linear-gradient(90deg, #0369a1, #0ea5e9, #38bdf8);
}
.vcard-body { padding: 26px 28px 22px; }

/* ── SIMILARITY BAR ───────────────────────────────── */
.sim-wrap { margin-top: 14px; }
.sim-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .10em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 7px;
}
.sim-score { color: #0284c7; font-size: 15px; letter-spacing: 0; font-weight: 700; }
.sim-bg {
    height: 8px;
    background: #e0f2fe;
    border-radius: 99px;
    overflow: hidden;
}
.sim-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #0369a1, #0ea5e9, #38bdf8);
}

/* ── VESSEL NAME ──────────────────────────────────── */
.vname {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    color: #0c4a6e;
    line-height: 1.18;
    margin-bottom: 4px;
}
.vship {
    font-size: 13px;
    color: #64748b;
    letter-spacing: .03em;
    margin-bottom: 18px;
}
.vship b { color: #0369a1; font-weight: 600; }

/* ── INFO GRID ────────────────────────────────────── */
.igrid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 18px;
}
.iitem {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 12px;
    padding: 11px 14px;
    transition: background .2s;
}
.iitem:hover { background: #e0f2fe; }
.ilbl {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 5px;
}
.ival { font-size: 14px; font-weight: 600; color: #0c4a6e; }
.ival.gold { color: #b45309; }
.ival.sky  { color: #0284c7; }
.ival.grn  { color: #059669; }

/* ── SECTION TAG ──────────────────────────────────── */
.stag {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 9px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.stag::after {
    content: "";
    flex: 1;
    height: 1px;
    background: #e0f2fe;
}

/* ── DESTINASI ROW ────────────────────────────────── */
.drow {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}
.ddot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #0ea5e9;
    flex-shrink: 0;
}
.dtxt { font-size: 13.5px; color: #334155; }

/* ── DIVIDER ──────────────────────────────────────── */
.sdiv {
    height: 1px;
    margin: 16px 0;
    border: none;
    background: linear-gradient(90deg, #bae6fd, #e0f2fe, transparent);
}

/* ── DESCRIPTION BOX ─────────────────────────────── */
.dbox {
    background: #f0f9ff;
    border-left: 3px solid #0ea5e9;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    font-size: 13.5px;
    color: #475569;
    line-height: 1.85;
    font-weight: 300;
    margin-bottom: 20px;
}

/* ── LINK BUTTON ──────────────────────────────────── */
.stLinkButton a {
    background: linear-gradient(135deg, #0369a1, #0284c7, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 11px 26px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    box-shadow: 0 4px 16px rgba(14,165,233,.28) !important;
    text-decoration: none !important;
}

/* ── EMPTY / LANDING ──────────────────────────────── */
.empty-wrap { text-align: center; padding: 80px 20px; }
.empty-icon { font-size: 54px; margin-bottom: 18px; }
.empty-h {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    color: #0c4a6e;
    margin-bottom: 10px;
}
.empty-p { font-size: 14px; color: #64748b; line-height: 1.8; }

/* ── FEATURE CARD (landing) ───────────────────────── */
.feat-card {
    text-align: center;
    padding: 22px;
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 18px;
    min-width: 140px;
    box-shadow: 0 2px 12px rgba(14,165,233,.08);
}
.feat-icon { font-size: 28px; margin-bottom: 10px; }
.feat-label {
    font-size: 11px;
    color: #0369a1;
    font-weight: 700;
    letter-spacing: .10em;
    text-transform: uppercase;
}
.feat-sub { font-size: 12px; color: #94a3b8; margin-top: 4px; }

/* ── FOOTER ───────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 50px 20px 24px;
    border-top: 1px solid #e0f2fe;
    margin-top: 60px;
}
.foot-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    color: #0c4a6e;
    margin-bottom: 6px;
}
.foot-sub {
    font-size: 11px;
    color: #94a3b8;
    letter-spacing: .10em;
    text-transform: uppercase;
}

/* ── RANK BADGE ─────────────────────────────────────── */
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #eff6ff;
    border: 2px solid #bfdbfe;
    font-size: 20px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚓ &nbsp; Sentence-BERT · Content-Based Filtering</div>
    <div class="hero-title">Sistem Rekomendasi<br>Paket Wisata Phinisi</div>
    <div class="hero-sub">Labuan Bajo · Nusa Tenggara Timur</div>
    <div class="hero-desc">
        Deskripsikan wisata impian Anda — sistem akan menemukan kapal phinisi
        terbaik menggunakan kecerdasan semantik Sentence-BERT
    </div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# QUERY INPUT PANEL
# ─────────────────────────────────────────
st.markdown('<div class="qpanel">', unsafe_allow_html=True)

st.markdown("""
<div class="qhead">🔍 &nbsp; Deskripsikan Wisata Impian Anda</div>
<div class="qhint">
    Tulis dalam Bahasa Indonesia atau Inggris. Semakin detail deskripsi Anda
    (tujuan, aktivitas, fasilitas, durasi), semakin relevan rekomendasinya.
</div>
""", unsafe_allow_html=True)

query_input = st.text_area(
    "query_area",
    placeholder=(
        "Contoh: Ingin liburan 3 hari di Pulau Komodo bersama keluarga, "
        "snorkeling di Pink Beach, kapal mewah ber-AC dengan kapasitas 10 orang..."
    ),
    height=115,
    key="qa"
)

st.markdown("""
<div class="example-label">💡 Contoh Query</div>
<div>
    <span class="example-chip">🤿 Snorkeling &amp; diving Komodo</span>
    <span class="example-chip">🏝️ Island hopping 5 hari keluarga</span>
    <span class="example-chip">👑 Honeymoon kapal VVIP mewah</span>
    <span class="example-chip">🐉 Trekking Pulau Komodo dragon</span>
    <span class="example-chip">📸 Sunset cruise Labuan Bajo</span>
    <span class="example-chip">🤿 Open trip budget snorkeling</span>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# FILTER ROW
# ─────────────────────────────────────────
c1, c2, c3 = st.columns([2.0, 0.85, 1.0])

with c1:
    kat_opts = ["Semua"] + sorted(df["kategori"].dropna().unique().tolist())
    kat = st.selectbox("FILTER KATEGORI", kat_opts)

with c2:
    topk = st.number_input("JUMLAH HASIL", min_value=1, max_value=20, value=5, step=1)

with c3:
    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    btn = st.button("🔍  Cari Rekomendasi", use_container_width=True)


# ─────────────────────────────────────────
# PROSES QUERY
# ─────────────────────────────────────────
if btn:
    if not query_input.strip():
        st.warning("⚠️  Silakan masukkan deskripsi wisata terlebih dahulu.")
    else:
        with st.spinner("🧠  Memproses Sentence-BERT embedding & cosine similarity..."):
            results = recommend(query_input.strip(), top_k=int(topk), kategori=kat)
        st.session_state.results    = results
        st.session_state.last_query = query_input.strip()


# ─────────────────────────────────────────
# RENDER HASIL
# ─────────────────────────────────────────
results    = st.session_state.results
last_query = st.session_state.last_query

# ── ADA HASIL ────────────────────────────
if results is not None and len(results) > 0:

    qd = last_query[:68] + ("..." if len(last_query) > 68 else "")
    st.markdown(f"""
    <div class="res-header">
        <div class="res-title">✦ &nbsp;{len(results)} Rekomendasi Terbaik</div>
        <div class="res-qtag">🔍 "{qd}"</div>
    </div>
    """, unsafe_allow_html=True)

    for i, row in results.iterrows():

        score     = float(row.get("skor", 0))
        score_pct = round(min(score * 100, 100), 1)
        medal     = rank_badge(i)
        kat_border_color, kat_bg_color, kat_icon = kategori_color(row["kategori"])

        # ── TOP BAR + OPEN CARD ────────────
        st.markdown(
            '<div class="vcard">'
            '<div class="vcard-top-bar"></div>'
            '<div class="vcard-body">',
            unsafe_allow_html=True
        )

        col_img, col_det = st.columns([1.0, 1.5])

        # ── KOLOM GAMBAR ──────────────────
        with col_img:
            # Rank medal
            st.markdown(f'<div class="rank-badge">{medal}</div>', unsafe_allow_html=True)

            # Gambar kapal
            try:
                st.image(row["image_url"], use_container_width=True)
            except Exception:
                st.markdown(
                    '<div style="height:180px;background:#e0f2fe;border-radius:16px;'
                    'display:flex;align-items:center;justify-content:center;'
                    'font-size:40px;">🚢</div>',
                    unsafe_allow_html=True
                )

            # Similarity bar
            st.markdown(f"""
            <div class="sim-wrap">
                <div class="sim-row">
                    <span>Skor Kesamaan</span>
                    <span class="sim-score">{score:.4f}</span>
                </div>
                <div class="sim-bg">
                    <div class="sim-fill" style="width:{score_pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── KOLOM DETAIL ─────────────────
        with col_det:

            # Kategori pill (inline style — paling aman di Streamlit)
            st.markdown(
                f'<div style="display:inline-flex;align-items:center;gap:5px;'
                f'padding:4px 14px;border-radius:99px;font-size:10px;font-weight:700;'
                f'letter-spacing:.12em;text-transform:uppercase;margin-bottom:12px;'
                f'background:{kat_bg_color};border:1.5px solid {kat_border_color};'
                f'color:{kat_border_color};">'
                f'{kat_icon} {row["kategori"]}</div>',
                unsafe_allow_html=True
            )

            # Nama paket & kapal
            st.markdown(
                f'<div class="vname">{row["nama_paket"]}</div>'
                f'<div class="vship">Kapal: <b>{row["nama_kapal"]}</b></div>',
                unsafe_allow_html=True
            )

            # Info grid 2×2
            st.markdown(f"""
            <div class="igrid">
                <div class="iitem">
                    <div class="ilbl">💰 Harga</div>
                    <div class="ival gold">{row["harga"]}</div>
                </div>
                <div class="iitem">
                    <div class="ilbl">⏱ Durasi</div>
                    <div class="ival sky">{row["durasi"]}</div>
                </div>
                <div class="iitem">
                    <div class="ilbl">👥 Kapasitas</div>
                    <div class="ival grn">{row["kapasitas"]}</div>
                </div>
                <div class="iitem">
                    <div class="ilbl">🏷 Kategori</div>
                    <div class="ival">{row["kategori"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Destinasi
            destinasi_text = str(row.get("destinasi", "-"))
            st.markdown(
                '<div class="stag">📍 Destinasi</div>'
                f'<div class="drow">'
                f'<div class="ddot"></div>'
                f'<div class="dtxt">{destinasi_text}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # ── FASILITAS (pakai st.write / columns agar aman) ──
            st.markdown('<div class="stag">✨ Fasilitas</div>', unsafe_allow_html=True)
            fasilitas_list = parse_chips(row.get("fasilitas", ""))
            if fasilitas_list:
                # Render sebagai st.columns kecil supaya tidak muncul HTML mentah
                n = len(fasilitas_list)
                ncols = min(n, 4)
                cols = st.columns(ncols)
                for idx, item in enumerate(fasilitas_list):
                    with cols[idx % ncols]:
                        st.markdown(
                            f'<div style="background:#f0f9ff;border:1px solid #bae6fd;'
                            f'border-radius:10px;padding:6px 10px;font-size:12px;'
                            f'color:#0369a1;margin-bottom:6px;text-align:center;">'
                            f'✦ {item}</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.markdown('<div style="font-size:13px;color:#94a3b8;">-</div>', unsafe_allow_html=True)

            # ── LAYANAN ─────────────────────────────────────────
            st.markdown('<div class="stag" style="margin-top:10px">🏝️ Layanan</div>', unsafe_allow_html=True)
            layanan_list = parse_chips(row.get("layanan", ""))
            if layanan_list:
                n2 = len(layanan_list)
                ncols2 = min(n2, 4)
                cols2 = st.columns(ncols2)
                for idx, item in enumerate(layanan_list):
                    with cols2[idx % ncols2]:
                        st.markdown(
                            f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;'
                            f'border-radius:10px;padding:6px 10px;font-size:12px;'
                            f'color:#059669;margin-bottom:6px;text-align:center;">'
                            f'◈ {item}</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.markdown('<div style="font-size:13px;color:#94a3b8;">-</div>', unsafe_allow_html=True)

        # ── DESKRIPSI (full width) ────────
        desc_text = str(row.get("deskripsi", "-"))
        st.markdown(
            '<div style="padding:0 4px 6px">'
            '<div class="sdiv"></div>'
            '<div class="stag">📋 Deskripsi Paket</div>'
            f'<div class="dbox">{desc_text}</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ── TOMBOL DETAIL ────────────────
        st.link_button("⚓  Lihat Detail Kapal", row["link"])

        # ── CLOSE CARD ───────────────────
        st.markdown("</div></div><div style='height:8px'></div>", unsafe_allow_html=True)


# ── HASIL KOSONG ─────────────────────────
elif results is not None and len(results) == 0:
    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-icon">⚓</div>
        <div class="empty-h">Tidak Ada Hasil Ditemukan</div>
        <div class="empty-p">
            Tidak ada kapal yang cocok dengan query dan filter yang dipilih.<br>
            Coba deskripsi berbeda atau ubah filter kategori.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── BELUM ADA QUERY (landing) ─────────────
else:
    st.markdown("""
    <div class="empty-wrap" style="padding:68px 20px 48px">
        <div class="empty-icon" style="font-size:60px">🚢</div>
        <div class="empty-h">Mulai Temukan Kapal Impian Anda</div>
        <div class="empty-p">
            Ketikkan deskripsi wisata di atas, lalu klik
            <strong style="color:#0284c7">Cari Rekomendasi</strong>.<br>
            Sistem akan menampilkan hasil lengkap beserta foto kapal,
            detail paket, fasilitas, layanan, dan skor kemiripan semantik.
        </div>
        <div style="margin-top:36px;display:flex;justify-content:center;gap:16px;flex-wrap:wrap;">
            <div class="feat-card">
                <div class="feat-icon">🧠</div>
                <div class="feat-label">Sentence-BERT</div>
                <div class="feat-sub">Pemahaman semantik</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon">📐</div>
                <div class="feat-label">Cosine Similarity</div>
                <div class="feat-sub">Ukur relevansi</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon">🎯</div>
                <div class="feat-label">Content-Based</div>
                <div class="feat-sub">Filter konten</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon">🚢</div>
                <div class="feat-label">Kapal Phinisi</div>
                <div class="feat-sub">Labuan Bajo</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="foot-title">⚓ Phinisi Recommendation System</div>
    <div class="foot-sub">
        Implementasi Sentence-BERT · Content-Based Filtering · Labuan Bajo
    </div>
</div>
""", unsafe_allow_html=True)
