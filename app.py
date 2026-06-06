import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HantaDetect — Clinical System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Global CSS — Midnight Navy & Electric Cyan Premium Theme ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-base:      #040814; /* Midnight Navy */
    --bg-panel:     #0a1128;
    --bg-card:      #0d1635;
    --bg-input:     #070d20;
    --border:       rgba(0, 240, 255, 0.15); /* Electric Cyan border */
    --border-hover: rgba(0, 240, 255, 0.4);
    --cyan:         #00f0ff;
    --cyan-light:   #8aebf1;
    --cyan-dim:     rgba(0, 240, 255, 0.3);
    --red:          #ff2a5f;
    --red-dim:      rgba(255, 42, 95, 0.2);
    --text-primary: #e2e8f0;
    --text-muted:   #64748b;
    --text-soft:    #94a3b8;
    --mono:         'DM Mono', monospace;
    --serif:        'DM Serif Display', serif;
    --sans:         'Inter', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    color: var(--text-primary);
    font-family: var(--sans);
}

/* Hide Streamlit chrome */
[data-testid="stToolbar"], footer, #MainMenu,
[data-testid="stDecoration"], [data-testid="stSidebar"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--cyan-dim); border-radius: 2px; }

/* ── App Container ── */
.block-container {
    max-width: 1200px !important;
    padding: 2rem 3rem 5rem !important;
}

/* ── Top Horizontal Navigation Bar (Styling st.radio) ── */
div[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: row;
    justify-content: center;
    background: var(--bg-panel);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 2.5rem;
    gap: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
div[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: transparent !important;
    padding: 0.6rem 1.8rem !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    cursor: pointer;
    transition: all 0.3s ease;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: var(--bg-input) !important;
    border-color: var(--cyan-dim) !important;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
}
/* Hide the radio circles */
div[data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
    display: none !important;
}
/* Style the text inside radio */
div[data-testid="stRadio"] div[class*="stMarkdown"] p {
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--cyan-light) !important;
    margin: 0 !important;
    font-weight: 500 !important;
}

/* ── Hero Header ── */
.hero-wrap {
    position: relative;
    padding: 1rem 0 2rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
}
.hero-eyebrow {
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 30px;
    height: 1px;
    background: var(--cyan);
}
.hero-title {
    font-family: var(--serif);
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 400;
    line-height: 1.1;
    color: var(--text-primary);
    margin-bottom: 1rem;
}
.hero-title em {
    font-style: italic;
    color: var(--cyan);
    text-shadow: 0 0 20px rgba(0, 240, 255, 0.3);
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-soft);
    font-weight: 300;
    max-width: 600px;
    line-height: 1.6;
}

/* ── Section Labels ── */
.section-label {
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 15px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Content Text ── */
.content-text {
    font-size: 1.05rem;
    line-height: 1.7;
    color: var(--text-soft);
    margin-bottom: 1.5rem;
}
.content-text strong {
    color: var(--cyan-light);
    font-weight: 500;
}

/* ── Card Panels ── */
.card-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.3s;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
.card-panel:hover { border-color: var(--border-hover); }

/* ── Form Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
    font-size: 0.95rem !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"]:focus-within > div > div {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 1px var(--cyan) !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    font-size: 0.75rem !important;
    font-family: var(--mono) !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #008f99 0%, #00f0ff 100%) !important;
    color: #040814 !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 240, 255, 0.4) !important;
}

/* ── Results Cards ── */
.result-positive {
    background: linear-gradient(135deg, rgba(255, 42, 95, 0.1), transparent);
    border: 1px solid rgba(255, 42, 95, 0.5);
    border-radius: 12px; padding: 2rem;
}
.result-negative {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.1), transparent);
    border: 1px solid rgba(0, 240, 255, 0.4);
    border-radius: 12px; padding: 2rem;
}

/* ── Model Compare Cards ── */
.model-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    border-left: 4px solid var(--border);
}
.model-card.winner {
    border-left: 4px solid var(--cyan);
    background: linear-gradient(to right, rgba(0, 240, 255, 0.05), transparent);
}
.model-title {
    font-family: var(--serif);
    font-size: 1.5rem;
    color: var(--cyan-light);
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('log_reg_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

@st.cache_data
def load_data():
    try:
        return pd.read_csv('hantavirus_detection_dataset.csv')
    except:
        return pd.DataFrame()

# ─── Navigation (Top Taskbar) ────────────────────────────────────────────────
st.markdown('<div style="margin-top: -2rem;"></div>', unsafe_allow_html=True)
page = st.radio(
    "Navigation",
    ["01. Introduction", "02. Visualizations", "03. Clinical Prediction"],
    horizontal=True,
    label_visibility="collapsed"
)

# ─── Helper for Plotly Dark Theme ────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='Inter'),
    xaxis=dict(gridcolor='rgba(0, 240, 255, 0.1)'),
    yaxis=dict(gridcolor='rgba(0, 240, 255, 0.1)')
)


if page == "01. Introduction":
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">Overview</div>
        <h1 class="hero-title">Understanding <em>Hantavirus</em></h1>
        <p class="hero-sub">
            An introduction to Hantavirus pathogenesis, the clinical dataset, and our machine learning architecture.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Disease Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content-text">
        <strong>Hantaviruses</strong> are a family of viruses spread mainly by rodents and can cause diverse disease syndromes in people worldwide. 
        Infection with any hantavirus can produce hantavirus disease. Hantaviruses in the Americas are known as "New World" hantaviruses and may cause Hantavirus Pulmonary Syndrome (HPS). Other hantaviruses, known as "Old World" hantaviruses, are found mostly in Europe and Asia and may cause Hemorrhagic Fever with Renal Syndrome (HFRS).
        <br><br>
        Early symptoms include fatigue, fever, and muscle aches. If left untreated, it can lead to severe respiratory and renal complications.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Dataset Information</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content-text">
        The predictive model in this application is built upon a highly specialized clinical dataset containing patient demographics, exposure history, and crucial laboratory panels (Hematology and Biochemistry). 
        By analyzing patterns in these clinical markers—such as white blood cell counts, platelet levels, and liver/kidney enzyme concentrations—we can establish correlations that are highly indicative of Hantavirus infection.
    </div>
    """, unsafe_allow_html=True)

    # ─── Model Selection Moved Here ───
    st.markdown('<div class="section-label">Model Selection Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-text">
        During development, we evaluated three distinct machine learning algorithms: <strong>Random Forest Classifier</strong>, <strong>K-Nearest Neighbors (KNN)</strong>, and <strong>Logistic Regression</strong>. Here is the scientific rationale for our final architecture:
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="model-card">
        <div class="model-title">Random Forest Classifier (Discarded)</div>
        <div class="content-text" style="margin-bottom:0;">
            The Random Forest model achieved a <strong>100% accuracy</strong> during testing. While this sounds ideal, it is actually highly problematic in a medical context. The model learned to instantly flag a patient as positive if the top 5 influential features (such as CRP) spiked heavily. In other words, if CRP was significantly above average, the model immediately declared the case positive. This indicates severe <em>overfitting</em> to extreme values. A reliable clinical model should not be blindly certain based merely on isolated extreme variables; it must capture the nuance of the overall physiological state.
        </div>
    </div>
    
    <div class="model-card">
        <div class="model-title">K-Nearest Neighbors / KNN (Discarded)</div>
        <div class="content-text" style="margin-bottom:0;">
            KNN was tested using 5 neighbors specifically to mitigate the severe class imbalance in our dataset (positive cases are significantly rarer than negative ones). This is a strong advantage for KNN: rather than making absolute rule-based decisions, it approaches the problem like a clinician comparing a new patient to the 5 historical patients with the most similar lab profiles. However, despite this benefit, we ultimately did not select KNN because it lacks the ability to output nuanced, continuous risk probabilities for clinical interpretation.
        </div>
    </div>
    
    <div class="model-card winner">
        <div class="model-title">Logistic Regression (Selected Architecture) 🏆</div>
        <div class="content-text" style="margin-bottom:0;">
            <strong>Why did we choose Logistic Regression?</strong> In the medical field, assessing the <em>probability</em> or risk percentage of an infection is vastly superior to a rigid "Positive/Negative" binary classification. Logistic Regression mathematically models the probability of the default class. It excels here because it evaluates the continuous risk across all lab panels, providing clinicians with a nuanced <strong>Probability Score</strong> (rather than a simple yes/no) that perfectly supports clinical decision-making.
        </div>
    </div>
    """, unsafe_allow_html=True)


elif page == "02. Visualizations":
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">Data Analytics</div>
        <h1 class="hero-title">Interactive <em>Visualizations</em></h1>
        <p class="hero-sub">
            Scientific explanation of predictive clinical features supported by interactive dataset distributions and correlation analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    if df.empty:
        st.warning("Dataset file not found.")
    else:
        # Create mapping for labels
        df['Diagnosis'] = df['Hantavirus_Positive'].map({0: 'Negative', 1: 'Positive'})
        color_map = {'Negative': '#00f0ff', 'Positive': '#ff2a5f'}

        # ─── Class Distribution ───
        st.markdown('<div class="section-label">Class Distribution Overview</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            <div class="card-panel" style="height:100%;">
                <h3 style="color:var(--cyan); font-family:var(--serif); margin-bottom:1rem;">Imbalance in Medical Data</h3>
                <div class="content-text" style="margin:0;">
                    This chart visualizes the ratio of Positive to Negative Hantavirus cases in our dataset. 
                    <br><br>
                    Like most real-world clinical datasets, there is a severe <strong>class imbalance</strong>. Positive cases are rare compared to negative ones. This imbalance is exactly why we initially considered KNN (to handle minority classes effectively) and ultimately chose Logistic Regression (to output probabilities rather than getting skewed by dominant classes).
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            fig_pie = px.pie(df, names='Diagnosis', title='Diagnosis Distribution',
                             color='Diagnosis', color_discrete_map=color_map, hole=0.6)
            fig_pie.update_layout(**PLOTLY_THEME, title_font=dict(color='#e2e8f0', family='DM Serif Display', size=24))
            st.plotly_chart(fig_pie, use_container_width=True)


        # ─── CRP & Creatinine Distributions ───
        st.markdown('<div class="section-label">Key Biomarker Distributions</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="content-text">
            <strong>CRP (C-Reactive Protein):</strong> A critical marker of systemic inflammation. Extremely high levels of CRP are frequently observed in positive cases due to the severe inflammatory cascade caused by the infection.<br>
            <strong>Creatinine:</strong> Elevated Creatinine levels indicate acute kidney injury, a hallmark of severe Hantavirus strains (HFRS). Notice how the positive cases shift towards higher values.
        </div>
        """, unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_crp = px.box(df, x='Diagnosis', y='CRP_mg/L', color='Diagnosis',
                             color_discrete_map=color_map, title='CRP Distribution by Diagnosis')
            fig_crp.update_layout(**PLOTLY_THEME, title_font=dict(color='#e2e8f0', family='DM Serif Display', size=20))
            st.plotly_chart(fig_crp, use_container_width=True)

        with col_c2:
            fig_creat = px.box(df, x='Diagnosis', y='Creatinine_mg/dL', color='Diagnosis',
                               color_discrete_map=color_map, title='Creatinine Distribution by Diagnosis')
            fig_creat.update_layout(**PLOTLY_THEME, title_font=dict(color='#e2e8f0', family='DM Serif Display', size=20))
            st.plotly_chart(fig_creat, use_container_width=True)


        # ─── Correlation Heatmap ───
        st.markdown('<div class="section-label">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="content-text">
            The heatmap below shows the correlation coefficient between numerical features. <br>
            A value close to <strong>1.0</strong> (Bright Cyan/White) indicates a strong positive correlation, while a value close to <strong>-1.0</strong> (Dark Navy/Red) indicates a strong negative correlation. 
            By looking at the row/column for <code>Hantavirus_Positive</code>, you can scientifically identify which lab panels (like WBC, CRP, ALT) most heavily influence the infection status.
        </div>
        """, unsafe_allow_html=True)

        # Select only numerical features for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        # Drop ID if it exists in numeric accidentally
        corr_matrix = numeric_df.corr()

        fig_corr = px.imshow(corr_matrix, 
                             text_auto='.2f', 
                             aspect="auto",
                             color_continuous_scale=[(0, '#040814'), (0.5, '#0d1635'), (1, '#00f0ff')],
                             title='Correlation Heatmap of Clinical Variables')
        
        fig_corr.update_layout(**PLOTLY_THEME, 
                               title_font=dict(color='#e2e8f0', family='DM Serif Display', size=24),
                               height=600)
        # Adjust text color for readability
        fig_corr.update_traces(textfont=dict(color='#e2e8f0'))
        
        st.plotly_chart(fig_corr, use_container_width=True)



elif page == "03. Clinical Prediction":
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">Diagnostic System</div>
        <h1 class="hero-title">Hanta<em>Detect</em> Engine</h1>
        <p class="hero-sub">
            Logistic Regression-based predictive analysis. Input patient demographic and laboratory parameters to generate an infection probability estimate.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, scaler, encoders = load_artifacts()

        # ─── SECTION 01 ───
        st.markdown('<div class="section-label">01 — Demographics & Exposure</div>', unsafe_allow_html=True)
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a: age = st.number_input("AGE (YEARS)", min_value=0, max_value=120, value=40)
        with col_b: gender = st.selectbox("GENDER", options=encoders['Gender'].classes_)
        with col_c: region = st.selectbox("REGION", options=encoders['Region'].classes_)
        with col_d: exposure_type = st.selectbox("EXPOSURE TYPE", options=encoders['Exposure_Type'].classes_)

        col_e, col_f, col_g, col_h = st.columns(4)
        with col_e: symptom_count = st.number_input("SYMPTOM COUNT", min_value=0, max_value=20, value=2)

        # ─── SECTION 02 ───
        st.markdown('<div class="section-label">02 — Hematology Panel</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: wbc_count = st.number_input("WBC COUNT (K/uL)", min_value=0.0, value=7.5, format="%.2f")
        with col2: platelet_count = st.number_input("PLATELET COUNT (K/uL)", min_value=0.0, value=250.0, format="%.1f")
        with col3: wbc_platelet_ratio = st.number_input("WBC / PLATELET RATIO", min_value=0.0, value=0.03, format="%.4f")

        # ─── SECTION 03 ───
        st.markdown('<div class="section-label">03 — Biochemistry Panel</div>', unsafe_allow_html=True)
        col3a, col3b, col3c = st.columns(3)
        with col3a:
            crp = st.number_input("CRP (mg/L)", min_value=0.0, value=5.0, format="%.2f")
            alt = st.number_input("ALT (U/L)",  min_value=0.0, value=30.0, format="%.2f")
        with col3b:
            ast = st.number_input("AST (U/L)",  min_value=0.0, value=30.0, format="%.2f")
            bun = st.number_input("BUN (mg/dL)", min_value=0.0, value=15.0, format="%.2f")
        with col3c:
            creatinine = st.number_input("CREATININE (mg/dL)", min_value=0.0, value=1.0, format="%.2f")
            creatinine_bun_ratio = st.number_input("CREATININE / BUN RATIO", min_value=0.0, value=0.06, format="%.4f")

        # ─── Predict ───
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

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("RUN PREDICTIVE ANALYSIS", type="primary")

        if predict_btn:
            prediction  = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            pct         = probability * 100
            neg_pct     = 100 - pct

            st.markdown('<div class="section-label" style="margin-top:2rem;">04 — Analysis Results</div>', unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <h2 style="color:var(--red); font-family:var(--serif); margin-bottom:1rem;">⚠ POSITIVE FOR HANTAVIRUS</h2>
                    <div class="content-text" style="color:var(--text-primary);">
                        The model indicates a <strong>High Risk</strong> of Hantavirus infection based on the clinical parameters provided.
                    </div>
                    <div style="font-family:var(--mono); font-size:2rem; color:var(--red); margin:1rem 0;">
                        Risk Probability: {pct:.1f}%
                    </div>
                    <div style="width:100%; height:6px; background:rgba(255,255,255,0.1); border-radius:3px;">
                        <div style="width:{pct:.1f}%; height:100%; background:var(--red); border-radius:3px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2 style="color:var(--cyan); font-family:var(--serif); margin-bottom:1rem;">✓ NEGATIVE FOR HANTAVIRUS</h2>
                    <div class="content-text" style="color:var(--text-primary);">
                        The model indicates a <strong>Low Risk</strong> of Hantavirus infection based on the clinical parameters provided.
                    </div>
                    <div style="font-family:var(--mono); font-size:2rem; color:var(--cyan); margin:1rem 0;">
                        Risk Probability: {pct:.1f}%
                    </div>
                    <div style="width:100%; height:6px; background:rgba(255,255,255,0.1); border-radius:3px;">
                        <div style="width:{pct:.1f}%; height:100%; background:var(--cyan); border-radius:3px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:3rem; padding-top:1.5rem; border-top:1px solid var(--border); font-family:var(--mono); font-size:0.75rem; color:var(--text-muted); line-height:1.6;">
            ⚠ DISCLAIMER — This system is developed strictly for educational and Data Science research purposes.<br>
            The generated output does NOT constitute a formal medical diagnosis and cannot replace the assessment of an experienced clinician.
        </div>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Model files (log_reg_model.pkl, scaler.pkl, encoders.pkl) not found. Please train the model first.")