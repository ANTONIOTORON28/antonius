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
# STRUKTUR CSV (final_dataset.csv):
#   nama_paket | nama_kapal | kategori | harga | durasi | kapasitas
#   destinasi  | fasilitas  | layanan  | deskripsi | image_url | link | content
#
#   kolom "content" = teks gabungan semua kolom, sudah lowercase/clean,
#   digunakan sebagai input embedding SBERT
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
# LOAD DATA & MODEL  (cached = cepat)
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_kapal_preprocessing.csv")
    df["content"] = df["content"].fillna("")
    return df

@st.cache_resource
def load_model():
    # Multilingual, ringan, akurat untuk B.Indonesia
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def build_corpus_embeddings(_model, contents: list):
    return _model.encode(contents, show_progress_bar=False)

df               = load_data()
model            = load_model()
corpus_emb       = build_corpus_embeddings(model, df["content"].tolist())

# ─────────────────────────────────────────
# FUNGSI REKOMENDASI SBERT
# ─────────────────────────────────────────
def recommend(query: str, top_k: int = 5, kategori: str = "Semua") -> pd.DataFrame:
    """
    1. Encode query pakai SBERT
    2. Cosine similarity vs semua dokumen
    3. Filter kategori (opsional)
    4. Return top_k diurutkan descending
    """
    q_emb  = model.encode([query])
    scores = cosine_similarity(q_emb, corpus_emb)[0]

    out = df.copy()
    out["skor"] = scores

    if kategori != "Semua":
        out = out[out["kategori"] == kategori]

    return out.sort_values("skor", ascending=False).head(top_k).reset_index(drop=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "results"    not in st.session_state: st.session_state.results    = None
if "last_query" not in st.session_state: st.session_state.last_query = ""

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def kategori_pill(k):
    ku = str(k).upper()
    if   "VVIP"     in ku: return f'<span class="pill pill-vvip">👑 {k}</span>'
    elif "VIP"      in ku: return f'<span class="pill pill-vip">💎 {k}</span>'
    elif "STANDARD" in ku or "STD" in ku: return f'<span class="pill pill-std">⚡ {k}</span>'
    else:                  return f'<span class="pill pill-oth">🚢 {k}</span>'

def rank_badge(i):
    medals = {0:"🥇", 1:"🥈", 2:"🥉"}
    return medals.get(i, f"#{i+1}")

def chips_html(raw: str, icon: str = "✦") -> str:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    return "".join(f'<span class="chip">{icon} {x}</span>' for x in items)

# ─────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── RESET / BASE ────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;}
.stApp{
    background:#030912;
    color:#dce8f5;
    font-family:'Outfit',sans-serif;
}
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none;}

/* ── SCROLLBAR ───────────────────────────────────── */
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#030912;}
::-webkit-scrollbar-thumb{
    background:linear-gradient(180deg,#0369a1,#06b6d4);
    border-radius:99px;
}

/* ── BACKGROUND GLOW ─────────────────────────────── */
.stApp::before{
    content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
    background:
        radial-gradient(ellipse 80% 50% at 8% 0%,  rgba(3,105,161,.14) 0%,transparent 55%),
        radial-gradient(ellipse 65% 40% at 92% 95%,rgba(6,182,212,.09) 0%,transparent 55%);
}

/* ── LAYOUT ──────────────────────────────────────── */
.block-container{padding:0 2.5rem 6rem!important;max-width:1340px!important;}
[data-testid="column"]{padding:0 10px!important;}

/* ════════════════════════════════════════════════════
   HERO
════════════════════════════════════════════════════ */
.hero{text-align:center;padding:52px 16px 0;margin-bottom:38px;}
.hero-badge{
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(6,182,212,.08);
    border:1px solid rgba(6,182,212,.22);
    border-radius:99px;padding:7px 22px;
    font-size:11px;font-weight:700;letter-spacing:.14em;
    text-transform:uppercase;color:#22d3ee;margin-bottom:20px;
}
.hero-title{
    font-family:'Playfair Display',serif;
    font-size:clamp(28px,4.5vw,56px);
    font-weight:700;line-height:1.12;
    background:linear-gradient(140deg,#f0f8ff 0%,#7dd3fc 45%,#22d3ee 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    margin-bottom:10px;
}
.hero-sub{
    font-size:13px;color:#3a6080;
    letter-spacing:.08em;text-transform:uppercase;margin-bottom:12px;
}
.hero-desc{
    font-size:15px;font-weight:300;color:#4a7090;
    line-height:1.75;max-width:540px;margin:0 auto;
}
.hero-line{
    width:90px;height:1px;margin:26px auto 0;
    background:linear-gradient(90deg,transparent,#0ea5e9 50%,transparent);
}

/* ════════════════════════════════════════════════════
   QUERY PANEL
════════════════════════════════════════════════════ */
.qpanel{
    background:rgba(4,12,24,.78);
    border:1px solid rgba(6,182,212,.14);
    border-radius:28px;padding:30px 34px 26px;
    margin-bottom:14px;backdrop-filter:blur(24px);
    box-shadow:0 8px 40px rgba(0,0,0,.40);
}
.qhead{
    font-size:11px;font-weight:700;
    letter-spacing:.16em;text-transform:uppercase;
    color:#22d3ee;margin-bottom:5px;
}
.qhint{
    font-size:13px;color:#2a5070;
    font-weight:300;margin-bottom:20px;line-height:1.6;
}
.example-label{
    font-size:11px;color:#1e4060;
    letter-spacing:.10em;text-transform:uppercase;
    font-weight:700;margin-bottom:9px;
}
.example-row{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:4px;}
.example-chip{
    padding:6px 13px;
    background:rgba(6,182,212,.06);
    border:1px solid rgba(6,182,212,.16);
    border-radius:10px;font-size:12px;color:#4a8aaa;
}

/* ── STREAMLIT TEXTAREA ───────────────────────────── */
.stTextArea textarea{
    background:rgba(2,8,18,.88)!important;
    border:1px solid rgba(6,182,212,.18)!important;
    border-radius:18px!important;color:#dce8f5!important;
    font-family:'Outfit',sans-serif!important;font-size:15px!important;
    line-height:1.7!important;padding:16px 20px!important;
    transition:border-color .25s,box-shadow .25s;
}
.stTextArea textarea:focus{
    border-color:rgba(6,182,212,.50)!important;
    box-shadow:0 0 0 3px rgba(6,182,212,.10)!important;
}
.stTextArea textarea::placeholder{color:#2a4560!important;}
.stTextArea label{display:none!important;}

/* ── SELECTBOX ────────────────────────────────────── */
.stSelectbox>div>div{
    background:rgba(2,8,18,.88)!important;
    border:1px solid rgba(6,182,212,.18)!important;
    border-radius:16px!important;color:#dce8f5!important;
    font-family:'Outfit',sans-serif!important;
}
.stSelectbox label{
    color:#2a5070!important;font-size:11px!important;
    font-weight:700!important;letter-spacing:.14em!important;
    text-transform:uppercase!important;
    font-family:'Outfit',sans-serif!important;
}

/* ── NUMBER INPUT ─────────────────────────────────── */
.stNumberInput>div>div{
    background:rgba(2,8,18,.88)!important;
    border:1px solid rgba(6,182,212,.18)!important;
    border-radius:16px!important;color:#dce8f5!important;
    font-family:'Outfit',sans-serif!important;
}
.stNumberInput label{
    color:#2a5070!important;font-size:11px!important;
    font-weight:700!important;letter-spacing:.14em!important;
    text-transform:uppercase!important;
    font-family:'Outfit',sans-serif!important;
}

/* ── BUTTON ───────────────────────────────────────── */
.stButton>button{
    background:linear-gradient(135deg,#0369a1 0%,#0284c7 60%,#0ea5e9 100%)!important;
    color:white!important;border:none!important;
    border-radius:16px!important;padding:13px 32px!important;
    font-family:'Outfit',sans-serif!important;
    font-weight:700!important;font-size:15px!important;
    letter-spacing:.05em!important;width:100%!important;
    box-shadow:0 4px 24px rgba(14,165,233,.30)!important;
    transition:all .25s!important;
}
.stButton>button:hover{
    box-shadow:0 6px 32px rgba(14,165,233,.48)!important;
    transform:translateY(-2px)!important;
}

/* ════════════════════════════════════════════════════
   RESULT HEADER
════════════════════════════════════════════════════ */
.res-header{
    display:flex;align-items:center;
    justify-content:space-between;flex-wrap:wrap;gap:10px;
    margin:36px 0 22px;
    padding-bottom:14px;
    border-bottom:1px solid rgba(255,255,255,.05);
}
.res-title{
    font-family:'Playfair Display',serif;
    font-size:22px;font-weight:700;color:#bdd8f0;
}
.res-qtag{
    display:inline-flex;align-items:center;gap:6px;
    background:rgba(6,182,212,.08);
    border:1px solid rgba(6,182,212,.20);
    border-radius:10px;padding:6px 14px;
    font-size:13px;color:#22d3ee;font-weight:500;
    max-width:460px;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap;
}

/* ════════════════════════════════════════════════════
   VESSEL CARD  (mirip screenshot)
════════════════════════════════════════════════════ */
.vcard{
    position:relative;
    background:linear-gradient(140deg,rgba(6,16,32,.97) 0%,rgba(3,9,20,.97) 100%);
    border-radius:28px;
    border:1px solid rgba(255,255,255,.06);
    overflow:hidden;margin-bottom:24px;
    transition:transform .32s cubic-bezier(.22,.68,0,1.15),box-shadow .32s;
    box-shadow:0 2px 4px rgba(0,0,0,.3),0 18px 52px rgba(0,0,0,.55);
}
.vcard::before{
    content:"";position:absolute;
    top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,#0ea5e9 40%,#22d3ee 60%,transparent);
    opacity:0;transition:opacity .32s;
}
.vcard:hover{transform:translateY(-5px);}
.vcard:hover::before{opacity:1;}
.vcard-body{padding:28px 30px 24px;}

/* ── IMAGE AREA ───────────────────────────────────── */
.img-wrap{position:relative;border-radius:18px;overflow:hidden;}
.img-wrap::after{
    content:"";position:absolute;inset:0;
    background:linear-gradient(
        135deg,rgba(6,182,212,.06) 0%,
        transparent 40%,rgba(0,0,0,.25) 100%
    );pointer-events:none;
}

/* ── RANK MEDAL ───────────────────────────────────── */
.rank-medal{
    position:absolute;top:12px;left:12px;z-index:10;
    width:40px;height:40px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:20px;
    background:rgba(5,13,26,.80);
    border:2px solid rgba(255,255,255,.12);
    backdrop-filter:blur(6px);
}

/* ── SIMILARITY BAR ───────────────────────────────── */
.sim-wrap{margin-top:14px;}
.sim-row{
    display:flex;justify-content:space-between;align-items:center;
    font-size:11px;font-weight:700;
    letter-spacing:.10em;text-transform:uppercase;
    color:#2a5070;margin-bottom:7px;
}
.sim-score{color:#22d3ee;font-size:15px;letter-spacing:0;font-weight:700;}
.sim-bg{height:6px;background:rgba(255,255,255,.05);border-radius:99px;overflow:hidden;}
.sim-fill{
    height:100%;border-radius:99px;
    background:linear-gradient(90deg,#0369a1,#0ea5e9,#22d3ee);
}

/* ── KATEGORI PILL ────────────────────────────────── */
.pill{
    display:inline-flex;align-items:center;gap:5px;
    padding:4px 14px;border-radius:99px;
    font-size:10px;font-weight:700;
    letter-spacing:.12em;text-transform:uppercase;
    margin-bottom:12px;
}
.pill-vvip{background:rgba(250,204,21,.10);border:1px solid rgba(250,204,21,.28);color:#facc15;}
.pill-vip {background:rgba(167,139,250,.10);border:1px solid rgba(167,139,250,.28);color:#a78bfa;}
.pill-std {background:rgba(52,211,153,.10);border:1px solid rgba(52,211,153,.28);color:#34d399;}
.pill-oth {background:rgba(6,182,212,.10);border:1px solid rgba(6,182,212,.28);color:#22d3ee;}

/* ── VESSEL NAME ──────────────────────────────────── */
.vname{
    font-family:'Playfair Display',serif;
    font-size:26px;font-weight:700;
    color:#f0f8ff;line-height:1.18;margin-bottom:4px;
}
.vship{
    font-size:12.5px;color:#2a5070;
    letter-spacing:.04em;margin-bottom:20px;
}
.vship b{color:#4a7090;font-weight:500;}

/* ── INFO GRID (2 x 2) ───────────────────────────── */
.igrid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:18px;}
.iitem{
    background:rgba(255,255,255,.025);
    border:1px solid rgba(255,255,255,.045);
    border-radius:13px;padding:11px 14px;
    transition:background .2s;
}
.iitem:hover{background:rgba(6,182,212,.06);}
.ilbl{
    font-size:9.5px;font-weight:700;
    letter-spacing:.12em;text-transform:uppercase;
    color:#1e4060;margin-bottom:4px;
}
.ival{font-size:14px;font-weight:600;color:#c8dff0;}
.ival.gold{color:#fbbf24;}
.ival.sky{color:#38bdf8;}
.ival.grn{color:#34d399;}

/* ── SECTION TAG ──────────────────────────────────── */
.stag{
    font-size:9.5px;font-weight:700;
    letter-spacing:.14em;text-transform:uppercase;
    color:#1e4060;margin-bottom:9px;
    display:flex;align-items:center;gap:8px;
}
.stag::after{content:"";flex:1;height:1px;background:rgba(255,255,255,.04);}

/* ── DESTINASI ROW ────────────────────────────────── */
.drow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:16px;}
.ddot{width:5px;height:5px;border-radius:50%;background:#0ea5e9;flex-shrink:0;}
.dtxt{font-size:13px;color:#4a7090;}

/* ── CHIPS (fasilitas / layanan) ─────────────────── */
.chips-wrap{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:14px;}
.chip{
    display:inline-flex;align-items:center;gap:5px;
    padding:5px 12px;border-radius:9px;
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.06);
    font-size:12px;font-weight:500;color:#5a8aaa;
    transition:all .2s;white-space:nowrap;
}
.chip:hover{background:rgba(6,182,212,.09);border-color:rgba(6,182,212,.22);color:#7dd3fc;}

/* ── DIVIDER ──────────────────────────────────────── */
.sdiv{
    height:1px;margin:16px 0;border:none;
    background:linear-gradient(90deg,rgba(6,182,212,.12),rgba(255,255,255,.03),transparent);
}

/* ── DESCRIPTION BOX ─────────────────────────────── */
.dbox{
    background:rgba(3,105,161,.05);
    border-left:2px solid rgba(14,165,233,.22);
    border-radius:0 10px 10px 0;
    padding:12px 16px;
    font-size:13px;color:#3a6080;
    line-height:1.85;font-weight:300;margin-bottom:20px;
}

/* ── LINK BUTTON ──────────────────────────────────── */
.stLinkButton a{
    background:linear-gradient(135deg,#0369a1,#0284c7,#0ea5e9)!important;
    color:white!important;border:none!important;
    border-radius:13px!important;padding:11px 26px!important;
    font-family:'Outfit',sans-serif!important;
    font-weight:600!important;font-size:13px!important;
    box-shadow:0 4px 20px rgba(14,165,233,.28)!important;
    text-decoration:none!important;
}
.stLinkButton a:hover{
    box-shadow:0 6px 28px rgba(14,165,233,.45)!important;
}

/* ── EMPTY / LANDING ──────────────────────────────── */
.empty-wrap{text-align:center;padding:80px 20px;}
.empty-icon{font-size:54px;margin-bottom:18px;opacity:.45;}
.empty-h{
    font-family:'Playfair Display',serif;
    font-size:26px;color:#2a4a6a;margin-bottom:10px;
}
.empty-p{font-size:14px;color:#1e3a5a;line-height:1.8;}

/* ── FOOTER ───────────────────────────────────────── */
.footer{
    text-align:center;padding:50px 20px 24px;
    border-top:1px solid rgba(255,255,255,.03);
    margin-top:60px;
}
.foot-title{
    font-family:'Playfair Display',serif;
    font-size:18px;color:#1e3a5a;margin-bottom:6px;
}
.foot-sub{font-size:11px;color:#0f2030;letter-spacing:.10em;text-transform:uppercase;}

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
<div class="example-row">
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

        # ── OPEN CARD ────────────────────
        st.markdown('<div class="vcard"><div class="vcard-body">', unsafe_allow_html=True)

        col_img, col_det = st.columns([1.0, 1.45])

        # ── KOLOM GAMBAR ──────────────────
        with col_img:

            # Gambar kapal
            st.markdown('<div class="img-wrap" style="position:relative">', unsafe_allow_html=True)
            st.image(row["image_url"], use_container_width=True)
            st.markdown(f"""
            <div class="rank-medal" style="position:absolute;top:12px;left:12px">
                {medal}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Similarity score bar
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

            # Nama paket
            st.markdown(
                kategori_pill(row["kategori"]) +
                f'<div class="vname">{row["nama_paket"]}</div>' +
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
            st.markdown(f"""
            <div class="stag">📍 Destinasi</div>
            <div class="drow">
                <div class="ddot"></div>
                <div class="dtxt">{row["destinasi"]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Fasilitas
            st.markdown(
                '<div class="stag">✨ Fasilitas</div>'
                '<div class="chips-wrap">' +
                chips_html(row["fasilitas"], "✦") +
                '</div>',
                unsafe_allow_html=True
            )

            # Layanan
            st.markdown(
                '<div class="stag">🏝️ Layanan</div>'
                '<div class="chips-wrap">' +
                chips_html(row["layanan"], "◈") +
                '</div>',
                unsafe_allow_html=True
            )

        # ── DESKRIPSI (full width) ────────
        st.markdown(f"""
        <div style="padding:0 4px 6px">
            <div class="sdiv"></div>
            <div class="stag">📋 Deskripsi Paket</div>
            <div class="dbox">{row["deskripsi"]}</div>
        </div>
        """, unsafe_allow_html=True)

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
            <strong style="color:#0ea5e9">Cari Rekomendasi</strong>.<br>
            Sistem akan menampilkan hasil lengkap seperti di bawah ini beserta<br>
            foto kapal, detail paket, fasilitas, layanan, dan skor kemiripan semantik.
        </div>

        <div style="
            margin-top:36px;
            display:flex;justify-content:center;gap:18px;flex-wrap:wrap;
        ">
            <div style="
                text-align:center;padding:20px 22px;
                background:rgba(6,182,212,.05);
                border:1px solid rgba(6,182,212,.10);
                border-radius:16px;min-width:145px;
            ">
                <div style="font-size:28px;margin-bottom:10px">🧠</div>
                <div style="font-size:11px;color:#1e4060;font-weight:700;
                    letter-spacing:.10em;text-transform:uppercase">Sentence-BERT</div>
                <div style="font-size:12px;color:#1a3050;margin-top:4px">Pemahaman semantik</div>
            </div>
            <div style="
                text-align:center;padding:20px 22px;
                background:rgba(6,182,212,.05);
                border:1px solid rgba(6,182,212,.10);
                border-radius:16px;min-width:145px;
            ">
                <div style="font-size:28px;margin-bottom:10px">📐</div>
                <div style="font-size:11px;color:#1e4060;font-weight:700;
                    letter-spacing:.10em;text-transform:uppercase">Cosine Similarity</div>
                <div style="font-size:12px;color:#1a3050;margin-top:4px">Ukur relevansi</div>
            </div>
            <div style="
                text-align:center;padding:20px 22px;
                background:rgba(6,182,212,.05);
                border:1px solid rgba(6,182,212,.10);
                border-radius:16px;min-width:145px;
            ">
                <div style="font-size:28px;margin-bottom:10px">🎯</div>
                <div style="font-size:11px;color:#1e4060;font-weight:700;
                    letter-spacing:.10em;text-transform:uppercase">Content-Based</div>
                <div style="font-size:12px;color:#1a3050;margin-top:4px">Filter konten</div>
            </div>
            <div style="
                text-align:center;padding:20px 22px;
                background:rgba(6,182,212,.05);
                border:1px solid rgba(6,182,212,.10);
                border-radius:16px;min-width:145px;
            ">
                <div style="font-size:28px;margin-bottom:10px">🚢</div>
                <div style="font-size:11px;color:#1e4060;font-weight:700;
                    letter-spacing:.10em;text-transform:uppercase">Kapal Phinisi</div>
                <div style="font-size:12px;color:#1a3050;margin-top:4px">Labuan Bajo</div>
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
