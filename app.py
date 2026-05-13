# =========================================================
# SISTEM REKOMENDASI PAKET WISATA KAPAL PHINISI
# FINAL PROFESSIONAL VERSION
# Sentence-BERT + Content-Based Filtering
# =========================================================

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Phinisi Recommendation System",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv("dataset_kapal_preprocessing.csv")

    # =====================================================
    # WAJIB ADA
    # =====================================================

    required_columns = {

        "nama_paket": "-",
        "nama_kapal": "-",
        "kategori": "-",
        "harga": "-",
        "durasi": "-",
        "kapasitas": "-",
        "destinasi": "-",
        "fasilitas": "-",
        "layanan": "-",
        "deskripsi": "-",
        "image_url": "",
        "link": "-",
        "content": ""
    }

    # =====================================================
    # AUTO CREATE KOLOM
    # =====================================================

    for col, default in required_columns.items():

        if col not in df.columns:
            df[col] = default

        df[col] = df[col].fillna(default)

    # =====================================================
    # AUTO CONTENT
    # =====================================================

    df["content"] = (

        df["nama_paket"].astype(str) + " " +
        df["kategori"].astype(str) + " " +
        df["fasilitas"].astype(str) + " " +
        df["layanan"].astype(str) + " " +
        df["destinasi"].astype(str) + " " +
        df["deskripsi"].astype(str)

    )

    return df


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )


# =========================================================
# EMBEDDING
# =========================================================
@st.cache_data
def create_embeddings(_model, contents):

    return _model.encode(
        contents,
        show_progress_bar=False
    )


# =========================================================
# LOAD
# =========================================================
df = load_data()

model = load_model()

embeddings = create_embeddings(
    model,
    df["content"].tolist()
)

# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================
def recommend(query, top_k=5, kategori="Semua"):

    query_embedding = model.encode([query])

    similarity = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    results = df.copy()

    results["score"] = similarity

    if kategori != "Semua":
        results = results[
            results["kategori"] == kategori
        ]

    results = results.sort_values(
        by="score",
        ascending=False
    )

    return results.head(top_k)


# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.stApp{
    background:#06111f;
    color:white;
}

section[data-testid="stSidebar"]{
    display:none;
}

.block-container{
    padding-top:2rem;
    padding-bottom:3rem;
    max-width:1400px;
}

/* HERO */

.hero{
    text-align:center;
    padding:30px 20px 40px 20px;
}

.hero-title{
    font-size:58px;
    font-weight:800;
    line-height:1.1;
    background:linear-gradient(90deg,#ffffff,#67e8f9);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero-sub{
    margin-top:14px;
    color:#94a3b8;
    font-size:18px;
}

/* SEARCH */

.search-box{
    background:#0b1727;
    padding:28px;
    border-radius:22px;
    border:1px solid rgba(255,255,255,.05);
    margin-bottom:25px;
}

/* CARD */

.card{
    background:#0b1727;
    border-radius:28px;
    overflow:hidden;
    border:1px solid rgba(255,255,255,.05);
    margin-bottom:24px;
    transition:.3s;
}

.card:hover{
    transform:translateY(-4px);
}

.title{
    font-size:30px;
    font-weight:700;
    margin-bottom:6px;
}

.badge{
    display:inline-block;
    padding:6px 14px;
    border-radius:999px;
    background:#082f49;
    color:#67e8f9;
    font-size:12px;
    margin-bottom:14px;
}

.info-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:12px;
    margin-top:20px;
}

.info-item{
    background:#08111d;
    padding:14px;
    border-radius:14px;
}

.info-label{
    font-size:11px;
    color:#64748b;
    margin-bottom:5px;
    text-transform:uppercase;
}

.info-value{
    font-size:15px;
    font-weight:600;
    color:white;
}

.chip{
    display:inline-block;
    padding:7px 12px;
    background:#08111d;
    border-radius:10px;
    margin:4px;
    color:#cbd5e1;
    font-size:12px;
}

.desc{
    background:#08111d;
    padding:16px;
    border-radius:14px;
    margin-top:18px;
    color:#cbd5e1;
    line-height:1.8;
}

.footer{
    text-align:center;
    margin-top:60px;
    color:#64748b;
    font-size:13px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">

    <div class="hero-title">
        Sistem Rekomendasi<br>
        Kapal Phinisi
    </div>

    <div class="hero-sub">
        Sentence-BERT · Content-Based Filtering · Labuan Bajo
    </div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# SEARCH PANEL
# =========================================================
st.markdown('<div class="search-box">', unsafe_allow_html=True)

query = st.text_area(

    "Deskripsikan wisata impian Anda",

    placeholder=
    "Contoh: Saya ingin kapal mewah untuk honeymoon 3 hari di Labuan Bajo dengan snorkeling dan sunset cruise",

    height=120
)

c1, c2, c3 = st.columns([2,1,1])

with c1:

    kategori_list = ["Semua"] + sorted(
        df["kategori"].unique().tolist()
    )

    kategori = st.selectbox(
        "Kategori",
        kategori_list
    )

with c2:

    top_k = st.number_input(
        "Jumlah Hasil",
        min_value=1,
        max_value=20,
        value=5
    )

with c3:

    st.markdown("<br>", unsafe_allow_html=True)

    search = st.button(
        "🔍 Cari Rekomendasi",
        use_container_width=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# SEARCH PROCESS
# =========================================================
if search:

    if query.strip() == "":

        st.warning(
            "Masukkan deskripsi wisata terlebih dahulu."
        )

    else:

        with st.spinner("Mencari rekomendasi terbaik..."):

            results = recommend(
                query,
                top_k,
                kategori
            )

        st.success(
            f"Ditemukan {len(results)} rekomendasi terbaik"
        )

        # =================================================
        # RESULT LOOP
        # =================================================
        for i, row in results.iterrows():

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            img_col, detail_col = st.columns([1,1.4])

            # =============================================
            # IMAGE
            # =============================================
            with img_col:

                image = row["image_url"]

                if image and image != "-":

                    st.image(
                        image,
                        use_container_width=True
                    )

                else:

                    st.image(
                        "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
                        use_container_width=True
                    )

            # =============================================
            # DETAIL
            # =============================================
            with detail_col:

                st.markdown(f"""
                <div style="padding:20px">

                    <div class="badge">
                        🚢 {row["kategori"]}
                    </div>

                    <div class="title">
                        {row["nama_paket"]}
                    </div>

                    <div style="
                        color:#94a3b8;
                        margin-bottom:12px;
                    ">
                        Kapal: {row["nama_kapal"]}
                    </div>

                </div>
                """, unsafe_allow_html=True)

                # =========================================
                # INFO GRID
                # =========================================
                st.markdown(f"""
                <div class="info-grid">

                    <div class="info-item">
                        <div class="info-label">
                            Harga
                        </div>
                        <div class="info-value">
                            {row["harga"]}
                        </div>
                    </div>

                    <div class="info-item">
                        <div class="info-label">
                            Durasi
                        </div>
                        <div class="info-value">
                            {row["durasi"]}
                        </div>
                    </div>

                    <div class="info-item">
                        <div class="info-label">
                            Kapasitas
                        </div>
                        <div class="info-value">
                            {row["kapasitas"]}
                        </div>
                    </div>

                    <div class="info-item">
                        <div class="info-label">
                            Similarity
                        </div>
                        <div class="info-value">
                            {round(row["score"]*100,2)}%
                        </div>
                    </div>

                </div>
                """, unsafe_allow_html=True)

                # =========================================
                # DESTINATION
                # =========================================
                st.markdown("### 📍 Destinasi")

                st.write(row["destinasi"])

                # =========================================
                # FACILITY
                # =========================================
                st.markdown("### ✨ Fasilitas")

                fasilitas = str(
                    row["fasilitas"]
                ).split(",")

                for f in fasilitas:

                    if f.strip():
                        st.markdown(
                            f'<span class="chip">{f}</span>',
                            unsafe_allow_html=True
                        )

                # =========================================
                # SERVICE
                # =========================================
                st.markdown("### 🏝️ Layanan")

                layanan = str(
                    row["layanan"]
                ).split(",")

                for l in layanan:

                    if l.strip():
                        st.markdown(
                            f'<span class="chip">{l}</span>',
                            unsafe_allow_html=True
                        )

                # =========================================
                # DESCRIPTION
                # =========================================
                st.markdown("### 📋 Deskripsi")

                st.markdown(f"""
                <div class="desc">
                    {row["deskripsi"]}
                </div>
                """, unsafe_allow_html=True)

                # =========================================
                # BUTTON
                # =========================================
                if row["link"] != "-":

                    st.link_button(
                        "⚓ Lihat Detail Kapal",
                        row["link"]
                    )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    ⚓ Phinisi Recommendation System <br>
    Sentence-BERT · Content-Based Filtering
</div>
""", unsafe_allow_html=True)
