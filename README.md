<div align="center">

# 🎓 Virtual Learning Evaluation Model

### Predicting Student Success in Virtual Learning Environments
#### Using Machine Learning on the OULAD Dataset

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-189AB4?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)

<br>

> *"Data science as a powerful equaliser in education — giving educators the right information, about the right students, at the right time."*

<br>

</div>

---

## 📖 Table of Contents

- [About The Project](#-about-the-project)
- [Dataset](#-dataset)
- [Key Success Parameters](#-key-success-parameters)
- [Project Pipeline](#-project-pipeline)
- [Models & Performance](#-models--performance)
- [Key Findings](#-key-findings)
- [Interactive Web App](#-interactive-web-app)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Team](#-team)

---

## 🌍 About The Project

In the modern era of digital education, Virtual Learning Environments (VLEs) have become the backbone of online and blended learning. Despite their widespread adoption, educational institutions often lack the analytical tools to objectively measure **how effective their VLE is**, **which students are at risk**, and **what specific factors drive student success or failure**.

This project addresses that gap by building a **machine learning-powered evaluation model** trained on the **Open University Learning Analytics Dataset (OULAD)** — one of the most comprehensive real-world educational datasets publicly available.

### What this project delivers

| Capability | Description |
|------------|-------------|
| 🔮 **Predict** | Each student's success probability before any exam takes place |
| 🚨 **Identify** | At-risk students weeks before they disengage or withdraw |
| 🔍 **Diagnose** | Whether underperformance is driven by engagement, academic, or demographic factors |
| 📊 **Measure** | VLE effectiveness at module and organisational level |
| 💡 **Recommend** | Evidence-based actions for educators and administrators |

---

## 📂 Dataset

**Source:** [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open-dataset)  
**Institution:** The Open University, United Kingdom  
**Published by:** Knowledge Media Institute (KMI)

The dataset contains anonymised records of **32,593 students** across **22 modules and presentations**, stored across 7 interrelated CSV files:

| File | Rows | Description |
|------|------|-------------|
| `courses.csv` | 22 | Module codes, presentation codes and module lengths |
| `assessments.csv` | 206 | Assessment types (TMA/CMA/Exam), due dates and weights |
| `vle.csv` | 6,364 | All learning materials available on the VLE platform |
| `studentInfo.csv` | 32,593 | Student demographics and final results |
| `studentRegistration.csv` | 32,593 | Registration and withdrawal dates per student |
| `studentAssessment.csv` | 173,912 | Every score submitted by every student |
| `studentVle.csv` | 10,655,280 | Every click made by every student on every VLE resource |

---

## 🔬 Key Success Parameters

We identified **15 key parameters** across three categories:

### 👤 Demographic Factors (8 Parameters)

| # | Parameter | What It Measures |
|---|-----------|-----------------|
| 1 | Gender | Whether being male or female influences success rates |
| 2 | Age Band | Whether student age group (0–35, 35–55, 55+) affects outcomes |
| 3 | Region | Whether geographic location plays a role in performance |
| 4 | Highest Education | Whether prior education level correlates with success |
| 5 | IMD Band | Whether socioeconomic deprivation level impacts outcomes |
| 6 | Disability | Whether having a disability affects a student's success rate |
| 7 | Previous Attempts | Whether retaking a module reduces chances of passing |
| 8 | Credits Studied | Whether taking on more credits increases risk of failure |

### 🖱️ VLE Engagement Factors (4 Parameters)

| # | Parameter | What It Measures |
|---|-----------|-----------------|
| 9 | Total VLE Clicks | Overall volume of interaction with the learning platform |
| 10 | Active Days | Number of distinct days the student logged into the VLE |
| 11 | Early Clicks | Clicks in the **first 30 days** — strongest early warning signal |
| 12 | Activity Type Diversity | How many different types of learning content the student explored |

### 📝 Assessment Performance Factors (3 Parameters)

| # | Parameter | What It Measures |
|---|-----------|-----------------|
| 13 | Average Assessment Score | Mean score across all TMA, CMA and Exam submissions |
| 14 | Average TMA Score | Coursework-specific average — most predictive of final result |
| 15 | Late Submission Rate | Proportion of assessments submitted after the deadline |

---

## 🗺️ Project Pipeline

```
Raw OULAD Data (7 CSV Files)
         │
         ▼
  Data Loading & Preview
         │
         ▼
  Exploratory Data Analysis
         │
         ▼
  Data Quality Check
  (Missing Values, Duplicates, Outliers)
         │
         ▼
  Data Preprocessing
  (Imputation, Encoding, Outlier Removal)
         │
         ▼
  Feature Engineering
  (15 Parameters from 7 Tables → Master DataFrame)
         │
         ▼
  Model Preparation
  (Train/Test Split 80/20 · Stratified)
         │
         ▼
  Model Training
  (Logistic Regression → Random Forest → XGBoost)
         │
         ▼
  Model Evaluation
  (Accuracy · F1 · AUC-ROC · Confusion Matrix)
         │
         ▼
  Feature Importance Analysis
         │
         ▼
  Student Segmentation
  (High Risk · Medium Risk · Low Risk)
         │
         ▼
  Insights & Recommendations
         │
         ▼
  Interactive Streamlit Web App
```

---

## 🤖 Models & Performance

We trained and compared three models of increasing complexity:

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression | ~72% | ~0.71 | ~0.78 |
| Random Forest | ~79% | ~0.78 | ~0.86 |
| XGBoost | ~81% | ~0.80 | ~0.88 |
| **Ensemble (Final)** | **~82%** | **~0.81** | **~0.89** |

The final prediction uses an **ensemble of Random Forest + XGBoost**, classifying students into:

| Risk Level | Probability | Action |
|------------|-------------|--------|
| 🟢 Low Risk | ≥ 65% | Monitor regularly |
| 🟡 Medium Risk | 40–64% | Proactive support recommended |
| 🔴 High Risk | < 40% | Immediate intervention required |

---

## 🔍 Key Findings

| # | Finding |
|---|---------|
| 1 | **Early engagement is the strongest predictor** — low VLE activity in the first 30 days is the clearest warning signal |
| 2 | **TMA scores outperform exam scores** — continuous assessment reflects sustained understanding better than final exams |
| 3 | **Behaviour predicts better than demographics** — what a student does on the VLE matters more than who they are |
| 4 | **Late submissions compound risk** — chronic late submissions strongly correlate with eventual withdrawal |
| 5 | **Socioeconomic background has measurable impact** — students from high-deprivation areas show lower success rates |
| 6 | **Repeat students need targeted plans** — re-enrolling without intervention does not improve outcomes |

---

## 🖥️ Interactive Web App

This project includes a fully interactive **Streamlit web application** that allows educators and administrators to:

- Select any student by ID and instantly view their predicted success probability
- See a full profile with demographic information and risk classification
- Explore performance radar charts, engagement timelines, and assessment breakdowns
- Read automatically generated insights and actionable recommendations
- View organisation-wide analytics including pass rates by module, risk distributions, and engagement vs. success scatter plots

### Run the App

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost plotly matplotlib seaborn

# Navigate to project folder
cd "your/project/folder"

# Run
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
📁 Virtual-Learning-Evaluation-Model/
│
├── 📓 Virtual_Learning_Evaluation_Model.ipynb   ← Main project notebook
├── 🖥️  app.py                                   ← Streamlit web application
├── 📄 README.md                                 ← You are here
│
└── 📁 data_set/
    ├── courses.csv
    ├── assessments.csv
    ├── vle.csv
    ├── studentInfo.csv
    ├── studentRegistration.csv
    ├── studentAssessment.csv
    └── studentVle.csv
```

> ⚠️ **Note:** The dataset CSV files are not included in this repository due to file size. Download them directly from [analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset) and place them in a `data_set/` folder.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Anaconda (recommended) or pip

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/Virtual-Learning-Evaluation-Model.git
cd Virtual-Learning-Evaluation-Model
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly streamlit jupyter
```

**3. Download the dataset**

Go to [https://analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset), download the ZIP file, extract it and place all CSV files in a folder called `data_set/` inside the project directory.

**4. Run the Jupyter Notebook**
```bash
jupyter notebook Virtual_Learning_Evaluation_Model.ipynb
```

**5. Run the Web App**
```bash
streamlit run app.py
```

---

## 👥 Team

| Name | Role |
|------|------|
| **Manar Khalid** ⭐ | Team Leader — Project management, pipeline oversight & quality review |
| Mohamed Elsakka | Data Engineer — Data loading, preprocessing & table merging |
| Mohamed Yahia Kamal | ML Engineer — Model building, training, tuning & evaluation |
| Shaimaa Essam | Feature Engineer — Feature design, encoding & target variable definition |
| Abd El-rahman Ali | Data Analyst — EDA, visualizations, insights & recommendations |
| Alaa Ibrahim Ali | Reporter — Notebook narrative, PowerPoint presentation & delivery |

---

## 🛠️ Built With

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

<div align="center">

*Virtual Learning Evaluation Model · OULAD Dataset · Graduation Project*

</div>
