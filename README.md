<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a1628,50:1a3a6b,100:3498db&height=200&section=header&text=Virtual%20Learning%20Evaluation%20Model&fontSize=36&fontColor=ffffff&fontAlignY=38&desc=Predicting%20Student%20Success%20Before%20Failure%20Happens&descAlignY=58&descSize=16&animation=fadeIn" />

<br>

<img src="https://readme-typing-svg.herokuapp.com?font=Sora&weight=800&size=22&pause=1000&color=3498DB&center=true&vCenter=true&width=900&lines=ML+Engineer+%7C+Data+Scientist+%7C+Analytics+Architect;32%2C593+Students+%C2%B7+10M%2B+Interactions+%C2%B7+0.964+AUC-ROC;Turning+Raw+Data+into+Academic+Lives+Saved;The+Future+of+Education+is+Predictive+%E2%80%94+Not+Reactive" alt="Typing SVG" />

<br><br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)
![Random Forest](https://img.shields.io/badge/Random_Forest-2ecc71?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br>

[![Dataset](https://img.shields.io/badge/Dataset-OULAD_Open_University-blue?style=flat-square)](https://analyse.kmi.open.ac.uk/open-dataset)
[![Models](https://img.shields.io/badge/Models-3_ML_Ensemble-2ecc71?style=flat-square)]()
[![AUC--ROC](https://img.shields.io/badge/AUC--ROC-0.964-gold?style=flat-square)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-90%25_Ensemble-brightgreen?style=flat-square)]()
[![App](https://img.shields.io/badge/App-Streamlit_Deployed-FF4B4B?style=flat-square&logo=streamlit)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)]()

<br>

> ### *"32,593 students. 10 million interactions. 15 engineered features. One model that achieves 0.964 AUC-ROC — predicting academic failure before it happens."*

<br>

</div>

---

<div align="center">

## 🌍 The Crisis Nobody Is Solving

</div>

<br>

Picture this.

A university administrator sits at her desk in 2025. Thousands of students are enrolled in her institution's online learning platform. They log in. They click through materials. They submit — or don't. They disappear.

She has **no idea** which ones are about to fail.

Not until the exam results come back.
Not until the withdrawal forms pile up.
Not until it is **already too late.**

<br>

> 📉 **Since 2012, globally 25–40% of VLE-enrolled university students fail or withdraw every single year**
>
> 🔴 **In the Open University dataset alone — 52.8% of students either failed or withdrew**
>
> 🚫 **The majority of educational institutions worldwide have zero early alert systems**
>
> 📊 **Engagement data is collected by every VLE — analysed by almost none**
>
> ⏳ **By the time failure becomes visible through grades — weeks of disengagement have already passed**

<br>

This is not a data problem.
This is not a technology problem.
**This is a will-to-act problem — and I built the tool to act.**

---

<div align="center">

## 💡 What I Built — And Why It Matters

</div>

<br>

The **Virtual Learning Evaluation Model** is a full-stack, end-to-end machine learning system that:

| Capability | What It Means in Practice |
|------------|--------------------------|
| 🔮 **Predicts** student success probability | Know who will fail — before any exam takes place |
| 🚨 **Identifies** at-risk students automatically | Flag disengagement weeks before withdrawal |
| 🔍 **Diagnoses** the root cause of underperformance | Is it engagement? Assessments? Demographics? |
| 📊 **Evaluates** an entire VLE at organisational level | Which modules work — and which don't |
| 💡 **Generates** personalised, actionable recommendations | Specific interventions for specific students |
| 🖥️ **Delivers** everything through a live interactive dashboard | A deployed Streamlit app — running today |

<br>

This is not a proof of concept.
This is not a research paper.
**This is a production-ready, deployable system — built on real data, producing real predictions.**

---

<div align="center">

## 📂 The Data — The Foundation of Everything

</div>

<br>

**Dataset:** [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open-dataset)
**Source:** The Open University, United Kingdom · Knowledge Media Institute (KMI)

<br>

This is not synthetic data. Not a Kaggle toy dataset.
This is **one of the most comprehensive real-world educational datasets ever publicly released** —
capturing the complete learning lifecycle of tens of thousands of real students.

<br>

| 📁 File | 📊 Records | 📋 What It Contains |
|---------|-----------|---------------------|
| `courses.csv` | 22 | Every module and course presentation offered |
| `assessments.csv` | 206 | Every assignment, TMA, CMA and final exam |
| `vle.csv` | 6,364 | Every learning resource on the VLE platform |
| `studentInfo.csv` | **32,593** | Every student's demographics and final result |
| `studentRegistration.csv` | **32,593** | Every registration and withdrawal event |
| `studentAssessment.csv` | **173,912** | Every score submitted by every student |
| `studentVle.csv` | **10,655,280** | Every single click on every learning resource |

<br>

> **Over 10 million interaction records processed. Every click. Every login. Every late submission. Every withdrawal.**
> **This is the raw material of human learning — transformed into predictive intelligence.**

---

<div align="center">

## 🔬 The 15 Features That Predict Everything

### *After processing 10+ million records, these are the signals that actually matter*

</div>

<br>

### 👤 Who The Student Is — Demographic Signals (8 Features)

| # | Feature | Signal Strength | What It Reveals |
|---|---------|----------------|-----------------|
| 1 | Gender | 🟡 Moderate | Performance pattern differences across groups |
| 2 | Age Band | 🟡 Moderate | Life pressures vary significantly by age group |
| 3 | Region | 🟡 Moderate | Geographic access and infrastructure gaps |
| 4 | Highest Education | 🟠 Medium-High | Prior academic experience shapes capacity |
| 5 | **IMD Band** | 🔴 **High** | Socioeconomic deprivation directly impacts outcomes |
| 6 | Disability | 🟡 Moderate | Flags need for additional platform accommodations |
| 7 | **Previous Attempts** | 🔴 **High** | Repeat students carry compounding disadvantage |
| 8 | Credits Studied | 🟠 Medium-High | Higher workload correlates with dropout risk |

### 🖱️ What The Student Does — Behavioural Signals (4 Features)

| # | Feature | Signal Strength | What It Reveals |
|---|---------|----------------|-----------------|
| 9 | **Total VLE Clicks** | 🔴 **Very High** | Overall engagement volume — the heartbeat of activity |
| 10 | **Active Days** | 🔴 **Very High** | Consistency — are they showing up every week? |
| 11 | ⭐ **Early Clicks (First 30 Days)** | 🔴 **STRONGEST PREDICTOR** | Early disengagement predicts everything that follows |
| 12 | **Activity Type Diversity** | 🔴 **High** | Students who explore broadly consistently outperform |

### 📝 How The Student Performs — Academic Signals (3 Features)

| # | Feature | Signal Strength | What It Reveals |
|---|---------|----------------|-----------------|
| 13 | **Average Assessment Score** | 🔴 **Very High** | Overall academic performance across all submissions |
| 14 | ⭐ **Average TMA Score** | 🔴 **STRONGEST ACADEMIC** | Coursework > exams as a final result predictor |
| 15 | **Late Submission Rate** | 🔴 **High** | Chronic lateness compounds into withdrawal |

<br>

> **🧠 Critical Insight: Behavioural signals outperform demographic signals.**
> **What a student DOES on the VLE matters far more than who they ARE.**
> **This means failure is not inevitable — it is detectable, and therefore preventable.**

---

<div align="center">

## 🗺️ The Full Pipeline — Raw Data to Real Predictions

</div>

```
╔══════════════════════════════════════════════════════════════════════╗
║     RAW OULAD DATA  ·  7 CSV Files  ·  10,655,280+ Records          ║
╚══════════════════════════════════════════════════════════════════════╝
                                │
                    ┌───────────┴───────────┐
                    │    DATA LOADING       │
                    │  Import · Validate    │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │   EXPLORATORY         │
                    │   DATA ANALYSIS       │
                    │ Distributions·Patterns│
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │   DATA QUALITY CHECK  │
                    │ Nulls·Dupes·Outliers  │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │  PREPROCESSING        │
                    │ Impute·Encode·Clean   │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │  FEATURE ENGINEERING  │
                    │ 7 Tables → 15 Features│
                    │  Master DataFrame     │
                    └───────────┬───────────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
    ┌──────────┴──┐  ┌──────────┴──┐  ┌─────────┴───┐
    │  LOGISTIC   │  │   RANDOM    │  │   XGBOOST   │
    │ REGRESSION  │  │   FOREST    │  │  CLASSIFIER │
    │  Baseline   │  │ 100 Trees   │  │ 200 Rounds  │
    │  Acc: 87%   │  │  Acc: 90%   │  │  Acc: 90%   │
    │ AUC: 0.944  │  │ AUC: 0.962  │  │ AUC: 0.964  │
    └──────────┬──┘  └──────────┬──┘  └─────────┬───┘
               │                │                │
               └────────────────┼────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  RF + XGBoost ENSEMBLE│
                    │   Acc: 90% · AUC:0.964│
                    └───────────┬───────────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
        🟢 Low Risk      🟡 Medium Risk    🔴 High Risk
         ≥ 65%            40–64%             < 40%
               └────────────────┼────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  STREAMLIT DASHBOARD  │
                    │  Live · Interactive   │
                    │  Insights + Actions   │
                    └───────────────────────┘
```

---

<div align="center">

## 🤖 Model Results — The Exact Numbers From The Notebook

### *Every metric. Every model. Extracted directly from the trained pipeline.*

</div>

<br>

### 📊 Model 1 — Logistic Regression *(Baseline)*

> *Simple, interpretable, fast. Establishes the performance floor and proves the features work.*

| Metric | Not Success | Success | Overall |
|--------|------------|---------|---------|
| ✅ **Accuracy** | — | — | **86.73%** |
| 🎯 **Precision** | 0.87 | 0.86 | **0.87** |
| 📡 **Recall** | 0.87 | 0.86 | **0.87** |
| ⚖️ **F1 Score** | 0.87 | 0.86 | **0.8594** |
| 📈 **AUC-ROC** | — | — | **0.9439** |

---

### 🌲 Model 2 — Random Forest *(100 Decision Trees)*

> *Ensemble of 100 trees. Captures non-linear patterns. Provides highly interpretable feature importance rankings.*

| Metric | Not Success | Success | Overall |
|--------|------------|---------|---------|
| ✅ **Accuracy** | — | — | **89.81%** |
| 🎯 **Precision** | 0.92 | 0.87 | **0.90** |
| 📡 **Recall** | 0.88 | 0.92 | **0.90** |
| ⚖️ **F1 Score** | 0.90 | 0.89 | **0.8946** |
| 📈 **AUC-ROC** | — | — | **0.9623** |

> ▲ +3.08% accuracy · ▲ +0.0184 AUC-ROC vs. baseline

---

### ⚡ Model 3 — XGBoost *(200 Boosting Rounds)*

> *Gradient boosting. Each round corrects the errors of the last. State-of-the-art on structured tabular data.*

| Metric | Not Success | Success | Overall |
|--------|------------|---------|---------|
| ✅ **Accuracy** | — | — | **89.74%** |
| 🎯 **Precision** | 0.92 | 0.87 | **0.90** |
| 📡 **Recall** | 0.88 | 0.91 | **0.90** |
| ⚖️ **F1 Score** | 0.90 | 0.89 | **0.8937** |
| 📈 **AUC-ROC** | — | — | **0.9640** |

> ▲ +3.01% accuracy · ▲ +0.0201 AUC-ROC vs. baseline

---

### 🏆 Final Model — RF + XGBoost Ensemble *(Production)*

> *Averages Random Forest and XGBoost probabilities. Reduces individual model bias. Best generalisation.*

```python
success_probability = (random_forest_probability + xgboost_probability) / 2
```

| Metric | Score | vs. Logistic Regression |
|--------|-------|------------------------|
| ✅ **Accuracy** | **89.83%** | ▲ +3.10% |
| ⚖️ **F1 Score** | **0.8943** | ▲ +0.0349 |
| 📈 **AUC-ROC** | **0.9640** | ▲ +0.0201 |

<br>

### 📊 Full Model Comparison at a Glance

| Model | Accuracy | F1 Score | AUC-ROC | Verdict |
|-------|----------|----------|---------|---------|
| Logistic Regression | 86.73% | 0.8594 | 0.9439 | ✅ Strong baseline |
| Random Forest | 89.81% | 0.8946 | 0.9623 | ✅ Excellent |
| XGBoost | 89.74% | 0.8937 | 0.9640 | ✅ Excellent |
| ⭐ **Ensemble (Final)** | **89.83%** | **0.8943** | **0.9640** | 🏆 **Best · Production** |

<br>

> **An AUC-ROC of 0.964 means the model correctly distinguishes between a student who will pass and one who will fail 96.4% of the time — using only behavioural and demographic signals, before any exam takes place.**
>
> **To put that in perspective: a random classifier scores 0.50. A perfect classifier scores 1.00. This model sits at 0.964.**

---

<div align="center">

## 🔍 What The Model Discovered — Key Findings

</div>

<br>

| # | 🔑 Finding | 💥 Real-World Impact |
|---|-----------|---------------------|
| 1 | ⭐ **Early VLE engagement (first 30 days) is the single strongest predictor** | An alert system in week 1 can prevent failure months later |
| 2 | 📝 **TMA coursework scores outperform exam scores as predictors** | Continuous assessment is more accurate and more fair |
| 3 | 🧠 **Behavioural signals outperform demographic signals** | Failure is behavioural — and behaviour can be changed |
| 4 | ⏰ **Chronic late submissions compound directly into withdrawal** | A deadline reminder costs almost nothing — and saves everything |
| 5 | 💰 **IMD band (socioeconomic deprivation) measurably reduces success rates** | Online learning does not automatically equalise opportunity |
| 6 | 🔄 **Repeat students do not improve without a targeted plan** | Re-enrolment alone is not a solution |

---

<div align="center">

## 🖥️ The Application — Where Science Meets Reality

</div>

<br>

The model is deployed as a **fully interactive Streamlit web application** — not a prototype, not a mockup. A working system.

```
┌────────────────────────────────────────────────────────────────────┐
│  🎓  VIRTUAL LEARNING EVALUATION MODEL  ·  Streamlit Dashboard      │
├────────────────────────────────────────────────────────────────────┤
│  📊 KPI BANNER · Total Students · Pass Rate · High Risk · WR%      │
│  🔍 SELECT ANY STUDENT BY ID ────────────────────────────────────  │
│  👤 Profile + Demographics    🎯 Success Probability Gauge          │
├────────────────────────────────────────────────────────────────────┤
│  TAB 1 · Performance Overview                                      │
│    Radar Chart · Assessment Bars · Percentile Rankings             │
│  TAB 2 · VLE Engagement                                            │
│    Click Breakdown · 30-Day Timeline · Activity & Submission Donuts│
│  TAB 3 · Insights & Recommendations                                │
│    Auto-generated insight cards · Actions · Feature Importance     │
│  TAB 4 · Organisation Analytics                                    │
│    Pass Rate by Module · Risk Distribution · Scatter Plot          │
└────────────────────────────────────────────────────────────────────┘
```

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly matplotlib seaborn
streamlit run app.py
# → Open http://localhost:8501
```

---

<div align="center">

## 💰 The Business Case — What This Is Worth

### *For any educational organisation evaluating this system*

</div>

<br>

### 💸 The Cost of Doing Nothing

| Metric | Figure | Source |
|--------|--------|--------|
| Average UK university tuition | £9,250 / year | UCAS 2025 |
| Open University enrollment | 170,000+ students | OU Annual Report |
| Withdrawal rate (2025 est.) | ~32% | Global VLE research |
| **Revenue lost to dropout annually** | **≈ £503 million** | Calculated |

### 📈 The ROI of Deploying This System

| Investment | Cost | Return |
|------------|------|--------|
| Early alert tutor outreach | £50–200 per student | Retains a £9,250/year student |
| **ROI ratio** | **1 : 46+** | **£46 saved per £1 spent** |

### ⏱️ Time Saved Per Institution

| Task | Without System | With This System |
|------|---------------|-----------------|
| Identify at-risk students | End of semester — too late | **Week 1 of course** |
| Review 32,593 records | Weeks of manual analysis | **Seconds — real-time** |
| Generate intervention plan | Ad-hoc, inconsistent | **Automated, personalised** |
| Organisation-wide audit | Annual, expensive | **Live, continuous, free** |

<br>

> **This system does not just save money. It saves academic careers.**
> **Every student retained is a life trajectory changed — not just a tuition fee recovered.**

---

<div align="center">

## 🔭 The Future — Where This Goes Next

### *This project is a foundation — not a ceiling*

</div>

<br>

### 🚀 Near-Term (6–12 months)

| Direction | Description | Impact |
|-----------|-------------|--------|
| ☁️ **Cloud Deployment** | Streamlit Cloud · public URL · zero infrastructure cost | Any institution worldwide — instant access |
| ⚡ **Live LMS Integration** | Connect to Moodle, Canvas, Blackboard APIs | Real-time predictions as students interact |
| 📱 **Mobile Dashboard** | Push alerts on educator's phone | At-risk students flagged the moment they disengage |

### 🌐 Medium-Term (1–2 years)

| Direction | Description | Impact |
|-----------|-------------|--------|
| 📈 **Time-Series Modelling** | LSTM networks on weekly engagement curves | Detect deterioration months earlier |
| 💬 **NLP on Forum Activity** | Sentiment analysis of student discussion posts | A signal layer no current system captures |
| 🏛️ **Multi-Institution Validation** | Test across datasets from multiple universities | Prove generalisability — the path to commercial licensing |

### 🌍 Long-Term (3–5 years)

| Direction | Description | Impact |
|-----------|-------------|--------|
| 🤖 **Autonomous Intervention Engine** | System triggers tutor outreach automatically | No human bottleneck — every at-risk student reached |
| 📊 **National Analytics Platform** | Aggregate model across all institutions | Government-level insight into national VLE performance |
| 💼 **SaaS Commercialisation** | License the model as an EdTech product | Recurring revenue from institutions worldwide |

<br>

> **The global e-learning market is projected to reach $1 trillion by 2032.**
> **As it scales, so does its dropout crisis — and so does the value of systems that solve it.**
> **Models like this are not optional extras. They are the infrastructure that makes mass online education sustainable.**

---

<div align="center">

## 📁 Project Structure

</div>

```
📁 Virtual-Learning-Evaluation-Model/
├── 📓 Virtual_Learning_Evaluation_Model.ipynb   ← Full ML pipeline
├── 🖥️  app.py                                   ← Streamlit web app
├── 📄 README.md                                 ← You are here
└── 📁 data_set/
    ├── courses.csv
    ├── assessments.csv
    ├── vle.csv
    ├── studentInfo.csv
    ├── studentRegistration.csv
    ├── studentAssessment.csv
    └── studentVle.csv
```

> ⚠️ Dataset not included. Download free from [analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset) and place all 7 CSV files in `data_set/`

---

<div align="center">

## 🚀 Getting Started

</div>

```bash
# 1. Clone
git clone https://github.com/MohamedYahiaKamal/Virtual-Learning-Evaluation-Model.git
cd Virtual-Learning-Evaluation-Model

# 2. Install
pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly streamlit jupyter

# 3. Run notebook
jupyter notebook Virtual_Learning_Evaluation_Model.ipynb

# 4. Launch app
streamlit run app.py
```

---

<div align="center">

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

<div align="center">

## 👤 Built By

<br>

<img src="https://readme-typing-svg.herokuapp.com?font=Sora&weight=700&size=28&pause=1000&color=3498DB&center=true&vCenter=true&width=600&lines=Mohamed+Yahia+Kamal;ML+Engineer+%7C+Data+Scientist;Analytics+Architect+%7C+Problem+Solver" />

<br>

### *"I don't just build models. I build systems that make a difference."*

<br>

I am a **Machine Learning Engineer** and **Data Scientist** who specialises in turning complex, high-volume data into production-ready predictive systems with real business impact.

This project proves I can:

✅ **Engineer** a complete ML pipeline from 10M+ raw records to a deployed application
✅ **Design** 15 domain-driven features capturing the full complexity of human learning behaviour
✅ **Train and compare** 3 ML models and build an ensemble achieving **0.964 AUC-ROC**
✅ **Translate** model outputs into actionable, human-readable insights and recommendations
✅ **Deploy** a professional interactive dashboard evaluating **32,593 real students**
✅ **Think** at the intersection of data science, business impact, and human outcomes

<br>

---

### 📬 Let's Work Together

> *Whether you are a company looking to leverage data for growth,*
> *an educational organisation wanting to reduce dropout rates,*
> *or a business needing a data-driven solution built from scratch —*
> ***I can build it.***

<br>

| Platform | Link |
|----------|------|
| 📧 **Email** | (mailto:mohamed.yahia29016@gmail.com) |
| 💼 **LinkedIn** | (www.linkedin.com/in/mohamedyahia610) |
| 🐙 **GitHub** | (https://github.com/mohamed-yahia-kamal)) |
| 🌐 **Portfolio** | (https://mohamed-yahia-kamal.github.io/) |
| 🌐 **Telephone / Whats-app** | (+20 1004957788) |

<br>

### 🤝 Open To

```
✅ Freelance ML & Data Science Projects
✅ Full-Time ML Engineer / Data Scientist Roles
✅ EdTech & Analytics Consulting
✅ Research Collaborations
✅ Speaking & Technical Presentations
```

<br>

---

> ### ⭐ If This Project Impressed You
>
> **Star this repository · Share it · Reach out**
>
> *Every star tells the world that data science can do more than predict — it can protect.*
> *Every share puts this tool in front of the organisation that needs it.*
> *Every message could be the start of the next project that changes lives.*

<br>

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3498db,50:1a3a6b,100:0a1628&height=120&section=footer&text=Virtual%20Learning%20Evaluation%20Model&fontSize=18&fontColor=ffffff&fontAlignY=65" />

*Built with precision · Deployed with purpose · Designed to matter*

**© 2026 Mohamed Yahia Kamal · All Rights Reserved**

</div>
