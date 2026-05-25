import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Virtual Learning Evaluation Model",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    background-attachment: fixed;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Header Banner ── */
.header-banner {
    background: linear-gradient(135deg, #0d2137 0%, #0a3d62 50%, #1a5276 100%);
    border: 1px solid rgba(52, 152, 219, 0.3);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(52,152,219,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.header-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(46,204,113,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.header-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
.header-title span { color: #3498db; }
.header-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.55);
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.header-badge {
    display: inline-block;
    background: rgba(52,152,219,0.15);
    border: 1px solid rgba(52,152,219,0.4);
    color: #5dade2;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Metric Cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: rgba(13,33,55,0.8);
    border: 1px solid rgba(52,152,219,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(52,152,219,0.5); }
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45);
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.35);
    margin-top: 0.3rem;
}
.metric-card .accent-bar {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 12px 12px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.01em;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(52,152,219,0.2);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Insight cards ── */
.insight-card {
    background: rgba(13,33,55,0.7);
    border-left: 3px solid;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.insight-card .insight-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.insight-card .insight-text {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.5;
}
.insight-green  { border-color: #2ecc71; }
.insight-green .insight-title  { color: #2ecc71; }
.insight-orange { border-color: #f39c12; }
.insight-orange .insight-title { color: #f39c12; }
.insight-red    { border-color: #e74c3c; }
.insight-red .insight-title    { color: #e74c3c; }
.insight-blue   { border-color: #3498db; }
.insight-blue .insight-title   { color: #3498db; }

/* ── Risk badge ── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.2rem;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.risk-low    { background: rgba(46,204,113,0.15); border: 1px solid rgba(46,204,113,0.5); color: #2ecc71; }
.risk-medium { background: rgba(243,156,18,0.15); border: 1px solid rgba(243,156,18,0.5); color: #f39c12; }
.risk-high   { background: rgba(231,76,60,0.15);  border: 1px solid rgba(231,76,60,0.5);  color: #e74c3c; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d2137 100%);
    border-right: 1px solid rgba(52,152,219,0.15);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMarkdown p {
    color: rgba(255,255,255,0.8) !important;
    font-family: 'Sora', sans-serif !important;
}
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid rgba(52,152,219,0.2);
    margin-bottom: 1.5rem;
}
.sidebar-logo .logo-icon { font-size: 2.5rem; }
.sidebar-logo .logo-title {
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 0.5rem;
    letter-spacing: -0.01em;
}
.sidebar-logo .logo-sub {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Plotly chart containers ── */
.chart-container {
    background: rgba(13,33,55,0.7);
    border: 1px solid rgba(52,152,219,0.15);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(13,33,55,0.5);
    padding: 4px;
    border-radius: 10px;
    border: 1px solid rgba(52,152,219,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.5);
    font-family: 'Sora', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(52,152,219,0.2) !important;
    color: #5dade2 !important;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(52,152,219,0.3), transparent);
    margin: 1.5rem 0;
}

/* ── Student info pill ── */
.info-pill {
    display: inline-block;
    background: rgba(52,152,219,0.1);
    border: 1px solid rgba(52,152,219,0.25);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #7fb3d3;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* Spinner color */
.stSpinner > div { border-top-color: #3498db !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG    = 'rgba(10,20,40,0)'
PAPER_BG   = 'rgba(10,20,40,0)'
GRID_COLOR = 'rgba(52,152,219,0.1)'
TEXT_COLOR = 'rgba(255,255,255,0.7)'
FONT_FAMILY = 'Sora, sans-serif'

def apply_theme(fig, height=320):
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=11),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            bgcolor='rgba(10,20,40,0.6)',
            bordercolor='rgba(52,152,219,0.2)',
            borderwidth=1,
            font=dict(size=10)
        )
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(size=10))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE  (cached so it only runs once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_prepare(
    courses_path, assessments_path, vle_path,
    student_info_path, student_reg_path,
    student_assess_path, student_vle_path
):
    # --- Load ---
    courses        = pd.read_csv(courses_path)
    assessments    = pd.read_csv(assessments_path)
    vle            = pd.read_csv(vle_path)
    student_info   = pd.read_csv(student_info_path)
    student_reg    = pd.read_csv(student_reg_path)
    student_assess = pd.read_csv(student_assess_path)
    student_vle    = pd.read_csv(student_vle_path)

    # --- Preprocessing ---
    assessments['date']    = assessments['date'].fillna(0)
    student_assess['score']= student_assess['score'].fillna(0)
    student_assess         = student_assess[student_assess['score'].between(0, 100)]

    # --- Master merge ---
    master = student_info.copy()
    master = master.merge(
        student_reg[['id_student','code_module','code_presentation','date_unregistration']],
        on=['id_student','code_module','code_presentation'], how='left'
    )
    assess_merged = student_assess.merge(
        assessments[['id_assessment','code_module','code_presentation','assessment_type','weight']],
        on='id_assessment', how='left'
    )
    avg_scores = assess_merged.groupby(
        ['id_student','code_module','code_presentation']
    )['score'].mean().reset_index()
    avg_scores.columns = ['id_student','code_module','code_presentation','avg_score']
    master = master.merge(avg_scores, on=['id_student','code_module','code_presentation'], how='left')

    vle_summary = student_vle.groupby(
        ['id_student','code_module','code_presentation']
    ).agg(total_clicks=('sum_click','sum'), active_days=('date','nunique')).reset_index()
    master = master.merge(vle_summary, on=['id_student','code_module','code_presentation'], how='left')
    master['total_clicks'] = master['total_clicks'].fillna(0)
    master['active_days']  = master['active_days'].fillna(0)

    # --- Encode ---
    categorical_cols = ['gender','region','highest_education','imd_band','age_band','disability']
    le = LabelEncoder()
    for col in categorical_cols:
        master[col] = master[col].fillna('Unknown')
        master[col + '_encoded'] = le.fit_transform(master[col])

    # --- VLE engagement features ---
    early_vle = student_vle[student_vle['date'] <= 30].groupby(
        ['id_student','code_module','code_presentation']
    )['sum_click'].sum().reset_index()
    early_vle.columns = ['id_student','code_module','code_presentation','early_clicks']
    master = master.merge(early_vle, on=['id_student','code_module','code_presentation'], how='left')
    master['early_clicks'] = master['early_clicks'].fillna(0)

    act_div = student_vle.merge(vle[['id_site','activity_type']], on='id_site', how='left') \
        .groupby(['id_student','code_module','code_presentation'])['activity_type'].nunique().reset_index()
    act_div.columns = ['id_student','code_module','code_presentation','activity_types_used']
    master = master.merge(act_div, on=['id_student','code_module','code_presentation'], how='left')
    master['activity_types_used'] = master['activity_types_used'].fillna(0)

    # --- Assessment features ---
    assess_with_dates = student_assess.merge(
        assessments[['id_assessment','code_module','code_presentation','date','assessment_type']],
        on='id_assessment', how='left'
    )
    assess_with_dates['is_late'] = (
        assess_with_dates['date_submitted'] > assess_with_dates['date']
    ).astype(int)
    late_rate = assess_with_dates.groupby(
        ['id_student','code_module','code_presentation']
    )['is_late'].mean().reset_index()
    late_rate.columns = ['id_student','code_module','code_presentation','late_submission_rate']
    master = master.merge(late_rate, on=['id_student','code_module','code_presentation'], how='left')
    master['late_submission_rate'] = master['late_submission_rate'].fillna(0)

    tma_scores = assess_with_dates[assess_with_dates['assessment_type'] == 'TMA'].groupby(
        ['id_student','code_module','code_presentation']
    )['score'].mean().reset_index()
    tma_scores.columns = ['id_student','code_module','code_presentation','avg_tma_score']
    master = master.merge(tma_scores, on=['id_student','code_module','code_presentation'], how='left')
    master['avg_tma_score'] = master['avg_tma_score'].fillna(0)

    # --- Target ---
    master['success'] = master['final_result'].map(
        {'Pass':1,'Distinction':1,'Fail':0,'Withdrawn':0}
    )

    # --- Risk group ---
    master['risk_group'] = 'Medium Risk'
    master.loc[(master['total_clicks'] < 100) & (master['avg_score'] < 50), 'risk_group'] = 'High Risk'
    master.loc[(master['total_clicks'] > 500) & (master['avg_score'] > 70), 'risk_group'] = 'Low Risk'

    return master, student_vle, assess_with_dates

@st.cache_resource(show_spinner=False)
def train_models(master):
    feature_cols = [
        'gender_encoded','region_encoded','highest_education_encoded',
        'imd_band_encoded','age_band_encoded','disability_encoded',
        'num_of_prev_attempts','studied_credits',
        'total_clicks','active_days','early_clicks','activity_types_used',
        'avg_score','avg_tma_score','late_submission_rate'
    ]
    X = master[feature_cols].fillna(0)
    y = master['success'].dropna()
    X = X.loc[y.index]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    xgb.fit(X_train, y_train)
    return rf, xgb, feature_cols, X_test, y_test

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: predict for one student row
# ─────────────────────────────────────────────────────────────────────────────
def predict_student(row, rf, xgb, feature_cols):
    x = row[feature_cols].fillna(0).values.reshape(1, -1)
    rf_prob  = rf.predict_proba(x)[0][1]
    xgb_prob = xgb.predict_proba(x)[0][1]
    ensemble = (rf_prob + xgb_prob) / 2
    return round(ensemble * 100, 1), round(rf_prob * 100, 1), round(xgb_prob * 100, 1)

def risk_label(prob):
    if prob >= 65: return "Low Risk",   "risk-low",   "🟢"
    if prob >= 40: return "Medium Risk","risk-medium","🟡"
    return             "High Risk",  "risk-high",  "🔴"

def color_for_prob(prob):
    if prob >= 65: return "#2ecc71"
    if prob >= 40: return "#f39c12"
    return "#e74c3c"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🎓</div>
        <div class="logo-title">VLE Evaluator</div>
        <div class="logo-sub">OULAD · Graduation Project</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Dataset Paths")
    st.markdown("<p style='font-size:0.75rem;color:rgba(255,255,255,0.4);'>Point to your OULAD CSV files</p>", unsafe_allow_html=True)

    base = st.text_input("Base folder path", value=r"data_set", help="Folder containing all 7 OULAD CSV files")

    use_demo = st.checkbox("⚡ Use demo/synthetic data", value=False,
                           help="Generate synthetic OULAD-like data for demonstration")

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("### ℹ️ About")
    st.markdown("""
    <p style='font-size:0.8rem;color:rgba(255,255,255,0.5);line-height:1.6;'>
    This app runs the full OULAD pipeline — preprocessing, feature engineering, 
    and ML models (Random Forest + XGBoost) — then lets you explore any student's 
    predicted success probability and actionable insights.
    </p>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div class="header-badge">Open University Learning Analytics Dataset</div>
    <div class="header-title">Virtual Learning<br><span>Evaluation Model</span></div>
    <div class="header-sub">Predict student success · Identify risk factors · Generate improvement insights</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def make_synthetic_data():
    """Generate synthetic OULAD-like data for demo purposes."""
    np.random.seed(42)
    n = 3000

    modules = ['AAA','BBB','CCC','DDD','EEE','FFF','GGG']
    presentations = ['2013J','2014J','2013B']
    genders = ['M','F']
    regions = ['East Anglian Region','Scotland','North Western Region','South East Region',
               'West Midlands Region','London Region','South Region','Wales']
    edu = ['HE Qualification','A Level or Equivalent','Lower Than A Level',
           'Post Graduate Qualification','No Formal quals']
    imd = ['0-10%','10-20%','20-30%','30-40%','40-50%',
           '50-60%','60-70%','70-80%','80-90%','90-100%']
    age_bands = ['0-35','35-55','55<=']

    code_module = np.random.choice(modules, n)
    code_pres   = np.random.choice(presentations, n)
    id_student  = np.random.choice(range(10000, 99999), n, replace=False)

    gender     = np.random.choice(genders, n)
    region     = np.random.choice(regions, n)
    highest_ed = np.random.choice(edu, n, p=[0.35,0.3,0.2,0.1,0.05])
    imd_band   = np.random.choice(imd, n)
    age_band   = np.random.choice(age_bands, n, p=[0.7, 0.25, 0.05])
    disability = np.random.choice(['Y','N'], n, p=[0.1, 0.9])
    prev_att   = np.random.choice([0,1,2,3], n, p=[0.75,0.15,0.07,0.03])
    credits    = np.random.choice([30,60,90,120,180,240], n)

    total_clicks     = np.random.exponential(600, n).clip(0, 5000).astype(int)
    active_days      = (total_clicks / 15 + np.random.normal(0,5,n)).clip(0, 200).astype(int)
    early_clicks     = (total_clicks * np.random.uniform(0.2, 0.6, n)).astype(int)
    act_types        = np.random.randint(1, 12, n)
    avg_score        = (40 + total_clicks/50 + np.random.normal(0,15,n)).clip(0, 100)
    avg_tma_score    = (avg_score + np.random.normal(0,8,n)).clip(0, 100)
    late_rate        = np.random.beta(1.5, 5, n)

    # Simulate final result with logical dependencies
    score_factor   = (avg_score - 50) / 50
    click_factor   = (total_clicks - 300) / 1000
    late_factor    = -late_rate * 0.5
    logit          = score_factor + click_factor * 0.8 + late_factor + np.random.normal(0, 0.3, n)
    prob_success   = 1 / (1 + np.exp(-logit * 2))
    success        = (prob_success > 0.5).astype(int)

    final_result = []
    for i in range(n):
        if success[i] == 1:
            final_result.append('Distinction' if avg_score[i] > 80 else 'Pass')
        else:
            final_result.append('Withdrawn' if early_clicks[i] < 50 else 'Fail')

    master = pd.DataFrame({
        'id_student': id_student,
        'code_module': code_module,
        'code_presentation': code_pres,
        'gender': gender,
        'region': region,
        'highest_education': highest_ed,
        'imd_band': imd_band,
        'age_band': age_band,
        'disability': disability,
        'num_of_prev_attempts': prev_att,
        'studied_credits': credits,
        'final_result': final_result,
        'total_clicks': total_clicks,
        'active_days': active_days,
        'early_clicks': early_clicks,
        'activity_types_used': act_types,
        'avg_score': avg_score.round(1),
        'avg_tma_score': avg_tma_score.round(1),
        'late_submission_rate': late_rate.round(3),
        'date_unregistration': [np.nan if r != 'Withdrawn' else np.random.randint(10,100)
                                 for r in final_result],
    })

    # Encode
    categorical_cols = ['gender','region','highest_education','imd_band','age_band','disability']
    le = LabelEncoder()
    for col in categorical_cols:
        master[col] = master[col].fillna('Unknown')
        master[col + '_encoded'] = le.fit_transform(master[col])

    master['success'] = master['final_result'].map(
        {'Pass':1,'Distinction':1,'Fail':0,'Withdrawn':0}
    )
    master['risk_group'] = 'Medium Risk'
    master.loc[(master['total_clicks'] < 100) & (master['avg_score'] < 50), 'risk_group'] = 'High Risk'
    master.loc[(master['total_clicks'] > 500) & (master['avg_score'] > 70), 'risk_group'] = 'Low Risk'

    return master

# Load REAL data correctly
with st.spinner("🔄 Loading and processing REAL dataset..."):

    import os

    # Automatically detect dataset folder
    possible_folders = [
        'data_set',
        'dataset',
        'Data_Set',
        'DATA_SET',
        '.',
    ]

    detected_base = None

    for folder in possible_folders:
        test_file = os.path.join(folder, 'courses.csv')
        if os.path.exists(test_file):
            detected_base = folder
            break

    # If nothing detected → use user input
    if detected_base is None:
        detected_base = base

    try:
        master, _, _ = load_and_prepare(
            os.path.join(detected_base, 'courses.csv'),
            os.path.join(detected_base, 'assessments.csv'),
            os.path.join(detected_base, 'vle.csv'),
            os.path.join(detected_base, 'studentInfo.csv'),
            os.path.join(detected_base, 'studentRegistration.csv'),
            os.path.join(detected_base, 'studentAssessment.csv'),
            os.path.join(detected_base, 'studentVle.csv'),
        )

        st.success(f"✅ REAL dataset loaded successfully from: {detected_base}")
        st.success(f"✅ Total students loaded: {len(master):,}")

    except Exception as e:

        st.error("❌ REAL dataset could not be loaded")
        st.error(str(e))

        st.warning("⚠️ Falling back to synthetic/demo data")
        master = make_synthetic_data()

with st.spinner("🤖 Training Random Forest & XGBoost models..."):
    rf, xgb_model, feature_cols, X_test, y_test = train_models(master)

# ─────────────────────────────────────────────────────────────────────────────
# ORGANISATION-LEVEL KPI BANNER
# ─────────────────────────────────────────────────────────────────────────────
total_students  = len(master)
pass_rate_org   = round(master['success'].mean() * 100, 1)
high_risk_count = (master['risk_group'] == 'High Risk').sum()
avg_clicks_org  = round(master['total_clicks'].mean(), 0)
withdrawal_rate = round((master['final_result'] == 'Withdrawn').mean() * 100, 1)

col1, col2, col3, col4, col5 = st.columns(5)
def kpi(col, label, value, sub, color):
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
        <div class="accent-bar" style="background:{color};"></div>
    </div>""", unsafe_allow_html=True)

kpi(col1, "Total Students",   f"{total_students:,}", "in dataset",                    "#3498db")
kpi(col2, "Org Pass Rate",    f"{pass_rate_org}%",   "Pass + Distinction",             "#2ecc71")
kpi(col3, "High Risk",        f"{high_risk_count:,}","students need intervention",     "#e74c3c")
kpi(col4, "Avg VLE Clicks",   f"{int(avg_clicks_org):,}", "per student",               "#9b59b6")
kpi(col5, "Withdrawal Rate",  f"{withdrawal_rate}%", "dropped out",                   "#f39c12")

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STUDENT SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🔍 Select a Student")
col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 1])

with col_sel1:
    student_ids = sorted(master['id_student'].unique())
    selected_id = st.selectbox("Student ID", student_ids, index=0,
                                help="Select a student to analyse their VLE performance and predictions")

# Filter to that student (may have multiple modules)
student_rows = master[master['id_student'] == selected_id]

with col_sel2:
    if len(student_rows) > 1:
        module_opts = student_rows['code_module'].unique().tolist()
        selected_module = st.selectbox("Module", module_opts)
        student_rows = student_rows[student_rows['code_module'] == selected_module]
    else:
        st.markdown("<br>", unsafe_allow_html=True)

with col_sel3:
    st.markdown("<br>", unsafe_allow_html=True)

student = student_rows.iloc[0]

# Predict
prob_ensemble, prob_rf, prob_xgb = predict_student(student, rf, xgb_model, feature_cols)
risk_txt, risk_cls, risk_icon = risk_label(prob_ensemble)
prob_color = color_for_prob(prob_ensemble)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT: Student Profile + Prediction + Tabs
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2])

# ── LEFT: Student Profile Card ──
with left_col:
    st.markdown('<div class="section-header">👤 Student Profile</div>', unsafe_allow_html=True)

    actual = student.get('final_result', 'N/A')
    actual_color = "#2ecc71" if actual in ['Pass','Distinction'] else "#e74c3c"

    st.markdown(f"""
    <div style="background:rgba(13,33,55,0.8);border:1px solid rgba(52,152,219,0.2);
                border-radius:14px;padding:1.5rem;">
        <div style="text-align:center;margin-bottom:1.2rem;">
            <div style="width:64px;height:64px;border-radius:50%;
                        background:linear-gradient(135deg,#1a5276,#2980b9);
                        display:flex;align-items:center;justify-content:center;
                        margin:0 auto 0.7rem;font-size:1.8rem;">👤</div>
            <div style="font-size:1.1rem;font-weight:700;color:#fff;">
                ID: <span style="font-family:'JetBrains Mono';color:#5dade2;">{selected_id}</span>
            </div>
            <div style="margin-top:0.4rem;">
                <span class="risk-badge {risk_cls}">{risk_icon} {risk_txt}</span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.82rem;">
            <div style="color:rgba(255,255,255,0.45);">Module</div>
            <div style="color:#fff;font-weight:500;">{student['code_module']}</div>
            <div style="color:rgba(255,255,255,0.45);">Presentation</div>
            <div style="color:#fff;font-weight:500;">{student['code_presentation']}</div>
            <div style="color:rgba(255,255,255,0.45);">Gender</div>
            <div style="color:#fff;font-weight:500;">{student['gender']}</div>
            <div style="color:rgba(255,255,255,0.45);">Age Band</div>
            <div style="color:#fff;font-weight:500;">{student['age_band']}</div>
            <div style="color:rgba(255,255,255,0.45);">Education</div>
            <div style="color:#fff;font-weight:500;font-size:0.75rem;">{student['highest_education']}</div>
            <div style="color:rgba(255,255,255,0.45);">IMD Band</div>
            <div style="color:#fff;font-weight:500;">{student['imd_band']}</div>
            <div style="color:rgba(255,255,255,0.45);">Disability</div>
            <div style="color:#fff;font-weight:500;">{'Yes' if student['disability']=='Y' else 'No'}</div>
            <div style="color:rgba(255,255,255,0.45);">Prev Attempts</div>
            <div style="color:#fff;font-weight:500;">{int(student['num_of_prev_attempts'])}</div>
            <div style="color:rgba(255,255,255,0.45);">Credits</div>
            <div style="color:#fff;font-weight:500;">{int(student['studied_credits'])}</div>
            <div style="color:rgba(255,255,255,0.45);">Actual Result</div>
            <div style="color:{actual_color};font-weight:700;">{actual}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauge ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎯 Success Probability</div>', unsafe_allow_html=True)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_ensemble,
        number={'suffix': '%', 'font': {'size': 36, 'family': 'JetBrains Mono', 'color': '#fff'}},
        delta={'reference': 50, 'increasing': {'color': '#2ecc71'}, 'decreasing': {'color': '#e74c3c'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': TEXT_COLOR, 'tickfont': {'size': 10}},
            'bar': {'color': prob_color, 'thickness': 0.25},
            'bgcolor': 'rgba(13,33,55,0.5)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40],  'color': 'rgba(231,76,60,0.15)'},
                {'range': [40, 65], 'color': 'rgba(243,156,18,0.15)'},
                {'range': [65, 100],'color': 'rgba(46,204,113,0.15)'}
            ],
            'threshold': {'line': {'color': prob_color, 'width': 3}, 'thickness': 0.75, 'value': prob_ensemble}
        }
    ))
    gauge.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
        height=220, margin=dict(l=10, r=10, t=10, b=0)
    )
    st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})

    # Model agreement
    st.markdown(f"""
    <div style="display:flex;gap:0.5rem;margin-top:0.2rem;">
        <div style="flex:1;background:rgba(13,33,55,0.7);border:1px solid rgba(52,152,219,0.2);
                    border-radius:8px;padding:0.6rem;text-align:center;">
            <div style="font-size:0.65rem;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:0.07em;">Random Forest</div>
            <div style="font-size:1.1rem;font-weight:700;color:#5dade2;font-family:'JetBrains Mono';">{prob_rf}%</div>
        </div>
        <div style="flex:1;background:rgba(13,33,55,0.7);border:1px solid rgba(52,152,219,0.2);
                    border-radius:8px;padding:0.6rem;text-align:center;">
            <div style="font-size:0.65rem;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:0.07em;">XGBoost</div>
            <div style="font-size:1.1rem;font-weight:700;color:#5dade2;font-family:'JetBrains Mono';">{prob_xgb}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT: Tabs ──
with right_col:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Overview",
        "🖱️ VLE Engagement",
        "💡 Insights & Recommendations",
        "🏢 Organisation Analytics"
    ])

    # ────────────────────────────────────────────
    # TAB 1 — Performance Overview
    # ────────────────────────────────────────────
    with tab1:
        r1c1, r1c2 = st.columns(2)

        # Radar chart of student vs. average
        with r1c1:
            st.markdown('<div class="section-header">📡 Student vs. Average Profile</div>', unsafe_allow_html=True)
            cats = ['Avg Score','TMA Score','VLE Clicks','Active Days','Early Clicks','Activity Types']
            # Normalize to 0-100
            def norm(val, col):
                mn, mx = master[col].min(), master[col].max()
                return round((val - mn) / (mx - mn + 1e-9) * 100, 1)

            stu_vals = [
                norm(student['avg_score'], 'avg_score'),
                norm(student['avg_tma_score'], 'avg_tma_score'),
                norm(student['total_clicks'], 'total_clicks'),
                norm(student['active_days'], 'active_days'),
                norm(student['early_clicks'], 'early_clicks'),
                norm(student['activity_types_used'], 'activity_types_used'),
            ]
            avg_vals = [
                norm(master['avg_score'].mean(), 'avg_score'),
                norm(master['avg_tma_score'].mean(), 'avg_tma_score'),
                norm(master['total_clicks'].mean(), 'total_clicks'),
                norm(master['active_days'].mean(), 'active_days'),
                norm(master['early_clicks'].mean(), 'early_clicks'),
                norm(master['activity_types_used'].mean(), 'activity_types_used'),
            ]

            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=avg_vals + [avg_vals[0]], theta=cats + [cats[0]],
                fill='toself', name='Org Average',
                line=dict(color='rgba(52,152,219,0.6)', width=1.5),
                fillcolor='rgba(52,152,219,0.08)'
            ))
            radar.add_trace(go.Scatterpolar(
                r=stu_vals + [stu_vals[0]], theta=cats + [cats[0]],
                fill='toself', name='This Student',
                line=dict(color=prob_color, width=2),
                fillcolor=prob_color.replace(')', ',0.15)').replace('rgb', 'rgba') if 'rgb' in prob_color
                          else f'rgba({int(prob_color[1:3],16)},{int(prob_color[3:5],16)},{int(prob_color[5:7],16)},0.15)'
            ))
            radar.update_layout(
                polar=dict(
                    bgcolor='rgba(13,33,55,0.5)',
                    radialaxis=dict(visible=True, range=[0,100], gridcolor=GRID_COLOR,
                                   tickfont=dict(size=8), color=TEXT_COLOR),
                    angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=9), color=TEXT_COLOR)
                ),
                paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                height=300, margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(bgcolor='rgba(10,20,40,0.6)', bordercolor='rgba(52,152,219,0.2)',
                            borderwidth=1, font=dict(size=10), orientation='h',
                            yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
            )
            st.plotly_chart(radar, use_container_width=True, config={'displayModeBar': False})

        # Assessment breakdown bar
        with r1c2:
            st.markdown('<div class="section-header">📝 Assessment Performance</div>', unsafe_allow_html=True)
            assess_fig = go.Figure()
            metrics_labels = ['Avg Score', 'TMA Score', 'Late Rate (×100)']
            stu_metrics = [
                round(student['avg_score'], 1),
                round(student['avg_tma_score'], 1),
                round(student['late_submission_rate'] * 100, 1)
            ]
            org_metrics = [
                round(master['avg_score'].mean(), 1),
                round(master['avg_tma_score'].mean(), 1),
                round(master['late_submission_rate'].mean() * 100, 1)
            ]
            assess_fig.add_trace(go.Bar(
                name='Org Average', x=metrics_labels, y=org_metrics,
                marker_color='rgba(52,152,219,0.5)', marker_line_color='rgba(52,152,219,0.8)',
                marker_line_width=1
            ))
            assess_fig.add_trace(go.Bar(
                name='This Student', x=metrics_labels, y=stu_metrics,
                marker_color=prob_color, opacity=0.85
            ))
            assess_fig.update_layout(
                barmode='group',
                xaxis_title=None, yaxis_title='Score / Rate',
            )
            apply_theme(assess_fig, height=300)
            st.plotly_chart(assess_fig, use_container_width=True, config={'displayModeBar': False})

        # Percentile bars
        st.markdown('<div class="section-header">📈 Student Percentile Rankings</div>', unsafe_allow_html=True)
        pct_cols = ['avg_score','avg_tma_score','total_clicks','active_days','early_clicks','late_submission_rate']
        pct_labels = ['Avg Score','TMA Score','Total Clicks','Active Days','Early Clicks','Late Rate']
        pct_vals, pct_colors = [], []
        for col in pct_cols:
            p = round((master[col] < student[col]).mean() * 100, 0)
            pct_vals.append(p)
            pct_colors.append('#e74c3c' if col == 'late_submission_rate' and p > 60 else
                              '#2ecc71' if p > 60 else
                              '#f39c12' if p > 35 else '#e74c3c')

        pct_fig = go.Figure(go.Bar(
            x=pct_vals, y=pct_labels, orientation='h',
            marker_color=pct_colors,
            text=[f'{v:.0f}th %ile' for v in pct_vals],
            textposition='outside', textfont=dict(size=10, color=TEXT_COLOR)
        ))
        pct_fig.add_vline(x=50, line_dash='dash', line_color='rgba(255,255,255,0.3)', line_width=1)
        pct_fig.update_xaxes(range=[0, 115])
        apply_theme(pct_fig, height=250)
        st.plotly_chart(pct_fig, use_container_width=True, config={'displayModeBar': False})

    # ────────────────────────────────────────────
    # TAB 2 — VLE Engagement
    # ────────────────────────────────────────────
    with tab2:
        e1, e2 = st.columns(2)

        with e1:
            st.markdown('<div class="section-header">🖱️ Click Breakdown</div>', unsafe_allow_html=True)
            click_labels = ['Total Clicks', 'Early Clicks (≤30d)', 'Late Clicks (>30d)']
            late_clicks = max(0, student['total_clicks'] - student['early_clicks'])
            click_vals = [student['total_clicks'], student['early_clicks'], late_clicks]
            click_fig = go.Figure(go.Bar(
                x=click_labels, y=click_vals,
                marker_color=['#3498db','#2ecc71','#9b59b6'],
                text=[f'{int(v):,}' for v in click_vals],
                textposition='outside', textfont=dict(color=TEXT_COLOR, size=11)
            ))
            org_avg_clicks = [master['total_clicks'].mean(), master['early_clicks'].mean(),
                              (master['total_clicks'] - master['early_clicks']).mean()]
            click_fig.add_trace(go.Scatter(
                x=click_labels, y=org_avg_clicks,
                mode='markers+lines', name='Org Avg',
                line=dict(color='rgba(255,255,255,0.4)', dash='dot', width=1.5),
                marker=dict(color='white', size=7)
            ))
            apply_theme(click_fig, height=280)
            st.plotly_chart(click_fig, use_container_width=True, config={'displayModeBar': False})

        with e2:
            st.markdown('<div class="section-header">📅 Engagement Intensity</div>', unsafe_allow_html=True)
            # Simulated timeline using student's known aggregate data
            days = list(range(1, 31))
            early = student['early_clicks']
            total = student['total_clicks']
            # Distribute clicks across simulated days using a curve
            weights_early = np.random.dirichlet(np.ones(30) * 2) * early
            timeline_fig = go.Figure()
            timeline_fig.add_trace(go.Bar(
                x=days, y=weights_early.round(0),
                name='Clicks per Day (est.)',
                marker_color='rgba(52,152,219,0.6)'
            ))
            timeline_fig.add_hline(y=early/30, line_dash='dash',
                                   line_color='rgba(46,204,113,0.6)', line_width=1.5,
                                   annotation_text="Daily avg", annotation_font_color='#2ecc71')
            timeline_fig.update_xaxes(title='Day of Course (first 30)')
            apply_theme(timeline_fig, height=280)
            st.plotly_chart(timeline_fig, use_container_width=True, config={'displayModeBar': False})

        e3, e4 = st.columns(2)
        with e3:
            st.markdown('<div class="section-header">🔢 Activity Types Used</div>', unsafe_allow_html=True)
            all_types = int(master['activity_types_used'].max())
            used_types = int(student['activity_types_used'])
            unused = all_types - used_types
            donut = go.Figure(go.Pie(
                values=[used_types, max(0, unused)],
                labels=['Used', 'Unused'],
                hole=0.6,
                marker_colors=['#3498db', 'rgba(52,152,219,0.15)'],
                textinfo='percent', textfont_size=11
            ))
            donut.add_annotation(
                text=f"<b>{used_types}</b><br><span style='font-size:10px'>of {all_types}</span>",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font=dict(color='white', family=FONT_FAMILY)
            )
            donut.update_layout(
                paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=True,
                legend=dict(bgcolor='rgba(10,20,40,0.5)', font=dict(size=10))
            )
            st.plotly_chart(donut, use_container_width=True, config={'displayModeBar': False})

        with e4:
            st.markdown('<div class="section-header">⏰ Submission Behaviour</div>', unsafe_allow_html=True)
            late_pct = round(student['late_submission_rate'] * 100, 1)
            on_time  = round(100 - late_pct, 1)
            sub_fig  = go.Figure(go.Pie(
                values=[on_time, late_pct],
                labels=['On Time', 'Late'],
                hole=0.6,
                marker_colors=['#2ecc71', '#e74c3c'],
                textinfo='percent+label', textfont_size=11
            ))
            sub_fig.add_annotation(
                text=f"<b>{late_pct}%</b><br><span style='font-size:9px'>late</span>",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font=dict(color='white', family=FONT_FAMILY)
            )
            sub_fig.update_layout(
                paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=True,
                legend=dict(bgcolor='rgba(10,20,40,0.5)', font=dict(size=10))
            )
            st.plotly_chart(sub_fig, use_container_width=True, config={'displayModeBar': False})

    # ────────────────────────────────────────────
    # TAB 3 — Insights & Recommendations
    # ────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">🔎 Automated Insights</div>', unsafe_allow_html=True)

        insights = []

        # Early engagement
        early_pct = (master['early_clicks'] < student['early_clicks']).mean() * 100
        if early_pct < 30:
            insights.append(('red', '⚠️ Low Early Engagement',
                f"This student clicked only {int(student['early_clicks'])} times in the first 30 days "
                f"(bottom {100-early_pct:.0f}% of all students). Early disengagement is the strongest "
                f"predictor of failure. Immediate tutor outreach is recommended."))
        elif early_pct > 70:
            insights.append(('green', '✅ Strong Early Engagement',
                f"This student showed excellent early engagement with {int(student['early_clicks'])} clicks "
                f"in the first 30 days (top {early_pct:.0f}%). This is a very positive success signal."))
        else:
            insights.append(('orange', '🔶 Moderate Early Engagement',
                f"Early engagement is average ({int(student['early_clicks'])} clicks). "
                f"Encouraging more activity in weeks 1–4 could improve outcomes."))

        # TMA score
        if student['avg_tma_score'] < 40:
            insights.append(('red', '⚠️ Critical TMA Performance',
                f"TMA average score is {student['avg_tma_score']:.1f}/100 — critically low. "
                f"Coursework grades are the strongest predictor of final results. "
                f"Targeted academic support is urgently needed."))
        elif student['avg_tma_score'] < 60:
            insights.append(('orange', '🔶 Below-Average TMA Score',
                f"TMA score of {student['avg_tma_score']:.1f}/100 is below the class average of "
                f"{master['avg_tma_score'].mean():.1f}. Additional coursework guidance is recommended."))
        else:
            insights.append(('green', '✅ Good TMA Performance',
                f"TMA score of {student['avg_tma_score']:.1f}/100 is strong "
                f"(class average: {master['avg_tma_score'].mean():.1f}). Keep it up."))

        # Late submissions
        if student['late_submission_rate'] > 0.5:
            insights.append(('red', '⚠️ High Late Submission Rate',
                f"{student['late_submission_rate']*100:.0f}% of submissions were late. "
                f"Chronic late submissions correlate strongly with withdrawal. "
                f"Recommend deadline management support and early reminders."))
        elif student['late_submission_rate'] > 0.2:
            insights.append(('orange', '🔶 Some Late Submissions',
                f"{student['late_submission_rate']*100:.0f}% late submission rate. "
                f"Minor concern — remind the student of upcoming deadlines proactively."))
        else:
            insights.append(('green', '✅ Excellent Submission Behaviour',
                f"Only {student['late_submission_rate']*100:.0f}% of submissions were late. "
                f"This student consistently meets deadlines."))

        # Activity diversity
        if student['activity_types_used'] < 4:
            insights.append(('orange', '🔶 Low Content Diversity',
                f"Student used only {int(student['activity_types_used'])} activity types. "
                f"Engaging with more content types (forums, quizzes, videos) is linked to better outcomes. "
                f"Recommend exploring different VLE resources."))
        else:
            insights.append(('green', '✅ Diverse Content Engagement',
                f"Student engaged with {int(student['activity_types_used'])} different activity types — "
                f"showing broad exploration of the VLE platform."))

        # Previous attempts
        if student['num_of_prev_attempts'] >= 2:
            insights.append(('red', '⚠️ Multiple Previous Attempts',
                f"This student has attempted this module {int(student['num_of_prev_attempts'])} time(s) before. "
                f"Repeat students have significantly higher dropout risk. "
                f"A personalised learning plan is strongly recommended."))
        elif student['num_of_prev_attempts'] == 1:
            insights.append(('orange', '🔶 Second Attempt',
                f"This is the student's second attempt. Targeted support based on previous failure reasons is advised."))

        # IMD / deprivation
        low_imd_bands = ['0-10%','10-20%','20-30%']
        if student.get('imd_band', '') in low_imd_bands:
            insights.append(('blue', 'ℹ️ Socioeconomic Consideration',
                f"Student is from IMD band {student['imd_band']} (high deprivation area). "
                f"Students from deprived areas are statistically more likely to face external pressures. "
                f"Consider additional wellbeing and financial support resources."))

        # Display insights
        for color, title, text in insights:
            st.markdown(f"""
            <div class="insight-card insight-{color}">
                <div class="insight-title">{title}</div>
                <div class="insight-text">{text}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="section-header">💡 Actionable Recommendations</div>', unsafe_allow_html=True)

        recs = []
        if student['early_clicks'] < master['early_clicks'].quantile(0.3):
            recs.append(("🚨 Immediate Action", "Send a personal tutor message this week to re-engage the student with the VLE platform.", "red"))
        if student['avg_tma_score'] < 55:
            recs.append(("📚 Academic Support", "Enrol student in supplementary TMA feedback sessions or peer study groups.", "orange"))
        if student['late_submission_rate'] > 0.3:
            recs.append(("⏰ Deadline Management", "Set up automated deadline reminders 7 and 3 days before each assessment.", "orange"))
        if student['activity_types_used'] < 4:
            recs.append(("🖥️ VLE Diversification", "Recommend the student explore forums, video resources and interactive quizzes.", "blue"))
        if student['num_of_prev_attempts'] >= 1:
            recs.append(("🔄 Repeat Student Plan", "Create a personalised learning plan addressing the specific weaknesses from previous attempts.", "red"))
        if prob_ensemble < 40:
            recs.append(("📞 Urgent Intervention", f"With only {prob_ensemble}% predicted success probability, escalate to student services immediately.", "red"))

        if not recs:
            st.markdown("""
            <div class="insight-card insight-green">
                <div class="insight-title">✅ No Immediate Actions Required</div>
                <div class="insight-text">This student is on track. Continue regular monitoring and encourage sustained engagement.</div>
            </div>""", unsafe_allow_html=True)
        else:
            for priority, text, color in recs:
                st.markdown(f"""
                <div class="insight-card insight-{color}">
                    <div class="insight-title">{priority}</div>
                    <div class="insight-text">{text}</div>
                </div>""", unsafe_allow_html=True)

        # Feature importance for this prediction
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🧠 What Drove This Prediction</div>', unsafe_allow_html=True)

        imp = pd.Series(rf.feature_importances_, index=feature_cols)
        readable = {
            'gender_encoded':'Gender','region_encoded':'Region',
            'highest_education_encoded':'Education Level','imd_band_encoded':'IMD Band',
            'age_band_encoded':'Age Band','disability_encoded':'Disability',
            'num_of_prev_attempts':'Prev Attempts','studied_credits':'Credits',
            'total_clicks':'Total Clicks','active_days':'Active Days',
            'early_clicks':'Early Clicks','activity_types_used':'Activity Types',
            'avg_score':'Avg Score','avg_tma_score':'TMA Score',
            'late_submission_rate':'Late Rate'
        }
        imp.index = [readable.get(i, i) for i in imp.index]
        imp = imp.sort_values()

        fi_fig = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation='h',
            marker=dict(
                color=imp.values,
                colorscale=[[0, '#1a3a5c'], [0.5, '#2980b9'], [1, '#3498db']],
                showscale=False
            ),
            text=[f'{v:.3f}' for v in imp.values],
            textposition='outside', textfont=dict(size=9, color=TEXT_COLOR)
        ))
        apply_theme(fi_fig, height=380)
        st.plotly_chart(fi_fig, use_container_width=True, config={'displayModeBar': False})

    # ────────────────────────────────────────────
    # TAB 4 — Organisation Analytics
    # ────────────────────────────────────────────
    with tab4:
        o1, o2 = st.columns(2)

        with o1:
            st.markdown('<div class="section-header">🏆 Pass Rate by Module</div>', unsafe_allow_html=True)
            mod_pass = master[master['final_result'].isin(['Pass','Distinction'])].groupby('code_module')['id_student'].count()
            mod_total = master.groupby('code_module')['id_student'].count()
            mod_rate = (mod_pass / mod_total * 100).reset_index()
            mod_rate.columns = ['Module', 'Pass Rate']
            mod_rate = mod_rate.sort_values('Pass Rate', ascending=True)

            mod_fig = go.Figure(go.Bar(
                x=mod_rate['Pass Rate'], y=mod_rate['Module'], orientation='h',
                marker_color=[
                    '#2ecc71' if v > 65 else '#f39c12' if v > 45 else '#e74c3c'
                    for v in mod_rate['Pass Rate']
                ],
                text=[f'{v:.1f}%' for v in mod_rate['Pass Rate']],
                textposition='outside', textfont=dict(size=10, color=TEXT_COLOR)
            ))
            mod_fig.add_vline(x=50, line_dash='dash', line_color='rgba(255,255,255,0.25)', line_width=1)
            mod_fig.update_xaxes(range=[0, 110])
            apply_theme(mod_fig, height=300)
            st.plotly_chart(mod_fig, use_container_width=True, config={'displayModeBar': False})

        with o2:
            st.markdown('<div class="section-header">🚦 Risk Group Distribution</div>', unsafe_allow_html=True)
            risk_counts = master['risk_group'].value_counts().reset_index()
            risk_counts.columns = ['Risk Group', 'Count']
            risk_colors = {'High Risk': '#e74c3c', 'Medium Risk': '#f39c12', 'Low Risk': '#2ecc71'}
            rg_fig = go.Figure(go.Pie(
                values=risk_counts['Count'],
                labels=risk_counts['Risk Group'],
                hole=0.55,
                marker_colors=[risk_colors.get(r, '#999') for r in risk_counts['Risk Group']],
                textinfo='percent+label', textfont_size=11
            ))
            rg_fig.update_layout(
                paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False
            )
            st.plotly_chart(rg_fig, use_container_width=True, config={'displayModeBar': False})

        o3, o4 = st.columns(2)
        with o3:
            st.markdown('<div class="section-header">👥 Pass Rate by Age Band</div>', unsafe_allow_html=True)
            age_pass = master[master['success']==1].groupby('age_band').size()
            age_total = master.groupby('age_band').size()
            age_rate = (age_pass / age_total * 100).reset_index()
            age_rate.columns = ['Age Band', 'Pass Rate']
            age_fig = px.bar(age_rate, x='Age Band', y='Pass Rate',
                             color='Pass Rate', color_continuous_scale=[[0,'#e74c3c'],[0.5,'#f39c12'],[1,'#2ecc71']],
                             text=[f'{v:.1f}%' for v in age_rate['Pass Rate']])
            age_fig.update_traces(textposition='outside', textfont_size=11)
            age_fig.update_coloraxes(showscale=False)
            apply_theme(age_fig, height=280)
            st.plotly_chart(age_fig, use_container_width=True, config={'displayModeBar': False})

        with o4:
            st.markdown('<div class="section-header">⚡ Engagement vs. Success</div>', unsafe_allow_html=True)
            sample = master.sample(min(500, len(master)), random_state=42)
            scatter = px.scatter(
                sample, x='total_clicks', y='avg_score',
                color='final_result',
                color_discrete_map={'Pass':'#2ecc71','Distinction':'#3498db',
                                    'Fail':'#e74c3c','Withdrawn':'#f39c12'},
                opacity=0.6, size_max=8
            )
            # Highlight selected student
            scatter.add_trace(go.Scatter(
                x=[student['total_clicks']], y=[student['avg_score']],
                mode='markers', name='Selected Student',
                marker=dict(color='white', size=14, symbol='star',
                            line=dict(color=prob_color, width=2))
            ))
            scatter.update_xaxes(title='Total VLE Clicks')
            scatter.update_yaxes(title='Avg Assessment Score')
            apply_theme(scatter, height=280)
            st.plotly_chart(scatter, use_container_width=True, config={'displayModeBar': False})

        # Final result distribution
        st.markdown('<div class="section-header">📊 Organisation-Wide Result Distribution</div>', unsafe_allow_html=True)
        result_counts = master['final_result'].value_counts().reset_index()
        result_counts.columns = ['Result', 'Count']
        result_color_map = {'Pass':'#2ecc71','Distinction':'#3498db','Fail':'#e74c3c','Withdrawn':'#f39c12'}
        dist_fig = px.bar(
            result_counts, x='Result', y='Count',
            color='Result',
            color_discrete_map=result_color_map,
            text='Count'
        )
        dist_fig.update_traces(textposition='outside', textfont_size=12)
        dist_fig.update_layout(showlegend=False)
        apply_theme(dist_fig, height=260)
        st.plotly_chart(dist_fig, use_container_width=True, config={'displayModeBar': False})

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1rem 0 0.5rem;">
    <span style="font-size:0.75rem;color:rgba(255,255,255,0.25);letter-spacing:0.05em;">
        VLE SUCCESS ANALYZER &nbsp;·&nbsp; OULAD DATASET &nbsp;·&nbsp; GRADUATION PROJECT &nbsp;·&nbsp;
        RANDOM FOREST + XGBOOST ENSEMBLE
    </span>
</div>
""", unsafe_allow_html=True)
