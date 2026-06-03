import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HantaDetect — Hantavirus Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Global CSS — Deep Navy × Amber Gold Medical Theme ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-base:      #0a0e1a;
    --bg-panel:     #0d1221;
    --bg-card:      #111827;
    --bg-input:     #0c1020;
    --border:       rgba(212,163,78,0.15);
    --border-hover: rgba(212,163,78,0.35);
    --gold:         #d4a34e;
    --gold-light:   #e8c27a;
    --gold-dim:     rgba(212,163,78,0.45);
    --red:          #e05252;
    --red-dim:      rgba(224,82,82,0.4);
    --green:        #4eb89a;
    --green-dim:    rgba(78,184,154,0.4);
    --text-primary: #e8e4db;
    --text-muted:   #6b7280;
    --text-soft:    #9ca3af;
    --mono:         'DM Mono', monospace;
    --serif:        'DM Serif Display', serif;
    --sans:         'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    color: var(--text-primary);
    font-family: var(--sans);
}

/* Hide Streamlit chrome */
[data-testid="stToolbar"], footer, #MainMenu,
[data-testid="stDecoration"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 2px; }

/* ── App Container ── */
.block-container {
    max-width: 1080px !important;
    padding: 0 2.5rem 5rem !important;
}

/* ── Top bar accent line ── */
.top-accent {
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--gold), var(--gold-light), transparent);
    margin-bottom: 0;
}

/* ── Hero Header ── */
.hero-wrap {
    position: relative;
    padding: 3.5rem 0 3rem;
    margin-bottom: 3rem;
    border-bottom: 1px solid var(--border);
}
.hero-eyebrow {
    font-family: var(--mono);
    font-size: 0.68rem;
    font-weight: 400;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 28px;
    height: 1px;
    background: var(--gold);
}
.hero-title {
    font-family: var(--serif);
    font-size: clamp(2.5rem, 5.5vw, 3.8rem);
    font-weight: 400;
    line-height: 1.0;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin-bottom: 1rem;
}
.hero-title em {
    font-style: italic;
    color: var(--gold);
}
.hero-sub {
    font-size: 0.88rem;
    color: var(--text-soft);
    font-weight: 300;
    max-width: 480px;
    line-height: 1.75;
}
.hero-meta {
    position: absolute;
    top: 3.5rem; right: 0;
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    text-align: right;
    line-height: 2.1;
    letter-spacing: 0.08em;
}
.hero-meta span { color: var(--gold-dim); }

/* ── Section Labels ── */
.section-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 12px;
    opacity: 0.85;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Card Panels ── */
.card-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    transition: border-color 0.3s;
}
.card-panel:hover { border-color: var(--border-hover); }
.card-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--gold), transparent);
    border-radius: 8px 0 0 8px;
    opacity: 0.4;
}

/* ── Number Input & Selectbox styling ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-input) !important;
    border: 1px solid rgba(212,163,78,0.18) !important;
    border-radius: 5px !important;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"]:focus-within > div > div {
    border-color: rgba(212,163,78,0.55) !important;
    box-shadow: 0 0 0 3px rgba(212,163,78,0.07) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    font-size: 0.68rem !important;
    font-family: var(--mono) !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Analyse Button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #c9973d 0%, #e8c27a 50%, #c9973d 100%) !important;
    background-size: 200% 100% !important;
    color: #0a0e1a !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.85rem 2.5rem !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: background-position 0.4s, transform 0.15s, box-shadow 0.2s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background-position: 100% 0 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(212,163,78,0.25) !important;
}

/* ── Result Cards ── */
.result-positive {
    background: linear-gradient(135deg, rgba(224,82,82,0.07), rgba(180,50,50,0.03));
    border: 1px solid rgba(224,82,82,0.35);
    border-radius: 8px;
    padding: 2rem 2.2rem;
    position: relative;
    overflow: hidden;
}
.result-positive::after {
    content: '⚠';
    position: absolute;
    right: 1.5rem; top: 1.2rem;
    font-size: 2.5rem;
    opacity: 0.06;
}
.result-negative {
    background: linear-gradient(135deg, rgba(78,184,154,0.07), rgba(50,150,120,0.03));
    border: 1px solid rgba(78,184,154,0.3);
    border-radius: 8px;
    padding: 2rem 2.2rem;
    position: relative;
    overflow: hidden;
}
.result-negative::after {
    content: '✓';
    position: absolute;
    right: 1.5rem; top: 1rem;
    font-size: 3.5rem;
    opacity: 0.07;
}
.result-status {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    opacity: 0.75;
}
.result-verdict {
    font-family: var(--serif);
    font-size: 2.2rem;
    font-weight: 400;
    line-height: 1.1;
    letter-spacing: -0.01em;
}
.col-pos { color: #e05252; }
.col-neg { color: #4eb89a; }
.col-gold { color: var(--gold); }

/* ── Probability Panel ── */
.prob-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 2rem 1.8rem;
    text-align: center;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 0.6rem;
}
.prob-label-top {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.prob-number {
    font-family: var(--serif);
    font-size: 3.8rem;
    font-weight: 400;
    line-height: 1;
}
.prob-unit {
    font-family: var(--mono);
    font-size: 1.2rem;
    vertical-align: super;
    opacity: 0.7;
}
.prob-bar-wrap {
    width: 100%;
    background: rgba(255,255,255,0.05);
    border-radius: 2px;
    height: 3px;
    overflow: hidden;
    margin-top: 0.4rem;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 2px;
}
.prob-split {
    font-family: var(--mono);
    font-size: 0.58rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* ── Info Row (stats) ── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
}
.stat-chip {
    flex: 1;
    background: rgba(212,163,78,0.05);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.stat-chip .s-val {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--gold-light);
}
.stat-chip .s-key {
    font-family: var(--mono);
    font-size: 0.57rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 3px;
}

/* ── Disclaimer ── */
.disclaimer {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: #3d4555;
    line-height: 1.8;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(212,163,78,0.06);
    letter-spacing: 0.03em;
}

/* ── Warning ── */
.warn-box {
    background: rgba(212,163,78,0.05);
    border: 1px solid rgba(212,163,78,0.25);
    border-radius: 8px;
    padding: 1.5rem 1.8rem;
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--gold);
    line-height: 1.7;
}

/* ── Column gap fix ── */
[data-testid="column"] { padding: 0 0.5rem !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Top Accent + Hero Header ────────────────────────────────────────────────
st.markdown('<div class="top-accent"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">Sistem Diagnostik Klinis</div>
    <h1 class="hero-title">Hanta<em>Detect</em></h1>
    <p class="hero-sub">
        Prediksi probabilitas infeksi Hantavirus berbasis Logistic Regression.
        Masukkan parameter laboratorium pasien untuk memperoleh estimasi risiko.
    </p>
    <div class="hero-meta">
        <span>MODEL</span> LOG_REG v1.0<br>
        <span>ENGINE</span> SKLEARN<br>
        <span>CLEAN</span> IQR-3 GROUPED<br>
        <span>BUILD</span> 2025.1
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Load Artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('log_reg_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

try:
    model, scaler, encoders = load_artifacts()

    # ─── SECTION 01 : Demografis & Paparan ──────────────────────────────────
    st.markdown('<div class="section-label">01 — Data Demografis & Riwayat Paparan</div>', unsafe_allow_html=True)
    

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        age = st.number_input("AGE (TAHUN)", min_value=0, max_value=120, value=40)
    with col_b:
        gender = st.selectbox("GENDER", options=encoders['Gender'].classes_)
    with col_c:
        region = st.selectbox("REGION / WILAYAH", options=encoders['Region'].classes_)
    with col_d:
        exposure_type = st.selectbox("TIPE PAPARAN", options=encoders['Exposure_Type'].classes_)

    col_e, col_f, col_g, col_h = st.columns(4)
    with col_e:
        symptom_count = st.number_input("SYMPTOM COUNT", min_value=0, max_value=20, value=2)

    st.markdown('</div>', unsafe_allow_html=True)

    # ─── SECTION 02 : Panel Hematologi ──────────────────────────────────────
    st.markdown('<div class="section-label">02 — Panel Hematologi</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        wbc_count          = st.number_input("WBC COUNT (K/uL)", min_value=0.0, value=7.5, format="%.2f")
    with col2:
        platelet_count     = st.number_input("PLATELET COUNT (K/uL)", min_value=0.0, value=250.0, format="%.1f")
    with col3:
        wbc_platelet_ratio = st.number_input("WBC / PLATELET RATIO", min_value=0.0, value=0.03, format="%.4f")

    st.markdown('</div>', unsafe_allow_html=True)

    # ─── SECTION 03 : Panel Biokimia ────────────────────────────────────────
    st.markdown('<div class="section-label">03 — Panel Biokimia & Fungsi Organ</div>', unsafe_allow_html=True)

    col3a, col3b, col3c = st.columns(3)
    with col3a:
        crp        = st.number_input("CRP (mg/L)", min_value=0.0, value=5.0, format="%.2f")
        alt        = st.number_input("ALT (U/L)",  min_value=0.0, value=30.0, format="%.2f")
    with col3b:
        ast        = st.number_input("AST (U/L)",  min_value=0.0, value=30.0, format="%.2f")
        bun        = st.number_input("BUN (mg/dL)", min_value=0.0, value=15.0, format="%.2f")
    with col3c:
        creatinine          = st.number_input("CREATININE (mg/dL)", min_value=0.0, value=1.0, format="%.2f")
        creatinine_bun_ratio = st.number_input("CREATININE / BUN RATIO", min_value=0.0, value=0.06, format="%.4f")

    st.markdown('</div>', unsafe_allow_html=True)

    # ─── Build Feature Vector ────────────────────────────────────────────────
    gender_encoded   = encoders['Gender'].transform([gender])[0]
    region_encoded   = encoders['Region'].transform([region])[0]
    exposure_encoded = encoders['Exposure_Type'].transform([exposure_type])[0]

    feature_names = [
        'Age', 'Gender', 'Region', 'Symptom_Count', 'Exposure_Type',
        'WBC_Count_K/uL', 'Platelet_Count_K/uL', 'CRP_mg/L', 'ALT_U/L',
        'AST_U/L', 'BUN_mg/dL', 'Creatinine_mg/dL',
        'WBC_Platelet_Ratio', 'Creatinine_BUN_Ratio'
    ]

    input_df = pd.DataFrame([[
        age, gender_encoded, region_encoded, symptom_count, exposure_encoded,
        wbc_count, platelet_count, crp, alt, ast, bun, creatinine,
        wbc_platelet_ratio, creatinine_bun_ratio
    ]], columns=feature_names)

    input_scaled = scaler.transform(input_df)

    # ─── Predict Button ──────────────────────────────────────────────────────
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🩺  Jalankan Analisis Prediktif", type="primary")

    if predict_btn:
        prediction  = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        pct         = probability * 100
        neg_pct     = 100 - pct

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">04 — Hasil Analisis Model</div>', unsafe_allow_html=True)

        r_col, p_col = st.columns([2, 1])

        with r_col:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <div class="result-status col-pos">☣ Status Infeksi Terdeteksi</div>
                    <div class="result-verdict col-pos">Positif<br>Hantavirus</div>
                    <div class="prob-bar-wrap" style="margin-top:1.2rem">
                        <div class="prob-bar-fill" style="width:{pct:.1f}%; background:linear-gradient(90deg,#e05252,#c03030)"></div>
                    </div>
                    <div class="stat-row">
                        <div class="stat-chip">
                            <div class="s-val">{pct:.1f}%</div>
                            <div class="s-key">Risiko Positif</div>
                        </div>
                        <div class="stat-chip">
                            <div class="s-val">{neg_pct:.1f}%</div>
                            <div class="s-key">Kepercayaan Negatif</div>
                        </div>
                        <div class="stat-chip">
                            <div class="s-val">HIGH</div>
                            <div class="s-key">Level Urgensi</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div class="result-status col-neg">✓ Status Infeksi</div>
                    <div class="result-verdict col-neg">Negatif<br>Hantavirus</div>
                    <div class="prob-bar-wrap" style="margin-top:1.2rem">
                        <div class="prob-bar-fill" style="width:{pct:.1f}%; background:linear-gradient(90deg,#4eb89a,#2e9c7e)"></div>
                    </div>
                    <div class="stat-row">
                        <div class="stat-chip">
                            <div class="s-val">{pct:.1f}%</div>
                            <div class="s-key">Risiko Positif</div>
                        </div>
                        <div class="stat-chip">
                            <div class="s-val">{neg_pct:.1f}%</div>
                            <div class="s-key">Kepercayaan Negatif</div>
                        </div>
                        <div class="stat-chip">
                            <div class="s-val">LOW</div>
                            <div class="s-key">Level Urgensi</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with p_col:
            prob_color = "#e05252" if prediction == 1 else "#4eb89a"
            gradient   = "linear-gradient(90deg,#e05252,#c03030)" if prediction == 1 else "linear-gradient(90deg,#4eb89a,#2e9c7e)"
            st.markdown(f"""
            <div class="prob-panel" style="border-color:{'rgba(224,82,82,0.3)' if prediction==1 else 'rgba(78,184,154,0.25)'}">
                <div class="prob-label-top">Probabilitas Infeksi</div>
                <div class="prob-number" style="color:{prob_color}">
                    {pct:.1f}<span class="prob-unit">%</span>
                </div>
                <div class="prob-bar-wrap" style="margin:0.6rem 0; width:100%">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{gradient}"></div>
                </div>
                <div class="prob-split">POS {pct:.1f}% · NEG {neg_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ─── Disclaimer ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠ DISCLAIMER — Sistem ini dikembangkan semata-mata untuk keperluan edukasi dan riset Data Science.<br>
        Output yang dihasilkan BUKAN merupakan diagnosis medis formal dan tidak dapat menggantikan penilaian klinisi berpengalaman.<br>
        Selalu konsultasikan hasil pemeriksaan kepada tenaga medis yang berkompeten sebelum mengambil keputusan klinis apapun.
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.markdown("""
    <div class="warn-box">
        ⚠ FILE MODEL TIDAK DITEMUKAN<br><br>
        Berkas <code>log_reg_model.pkl</code>, <code>scaler.pkl</code>, dan/atau <code>encoders.pkl</code>
        belum ada di direktori yang sama dengan <code>app.py</code>.<br><br>
        Jalankan terlebih dahulu: <code>python train_model.py</code>
    </div>
    """, unsafe_allow_html=True)