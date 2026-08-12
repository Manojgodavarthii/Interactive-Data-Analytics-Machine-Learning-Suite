# 📊 End-to-End Interactive Data Analytics & Machine Learning Suite

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An enterprise-ready, interactive web application built with **Python** and **Streamlit** for seamless data governance, exploratory data analysis (EDA), statistical validation, dynamic reporting, and live predictive machine learning.

This application allows users to upload custom datasets, evaluate data health, perform transformations, generate automated visual storyboards, and train real-time machine learning models with custom inference testing.

---

## ✨ Key Features

### 1. 🧹 Data Cleaning & Governance
* **Data Quality Index (DQI):** Automated assessment of completeness, uniqueness, data type validity, and outlier ratios.
* **Smart Data Cleaning:** Handle missing values, remove duplicate records, and perform column type casting.
* **Audit Trail & Version Control:** Reversible state management tracking step-by-step dataset modifications.

### 2. 📊 Exploratory Visual Analytics
* **Automated Dashboarding:** Dynamic generation of distribution plots, bar charts, box plots, and heatmaps.
* **Correlation Engine:** Parametric (Pearson) and non-parametric (Spearman) matrix calculations.
* **Feature & Column Explorer:** Deep-dive statistics on cardinality, skewness, and variance across continuous and categorical variables.

### 3. 🤖 AI Insights & Reporting
* **Natural Language Storytelling:** Executive-level KPI summaries and structured findings.
* **Automated Exporter:** One-click conversion of dashboard outputs into printable reports.

### 4. 🔮 Interactive Machine Learning & Predictive Inference
* **Automatic Problem Type Detection:** Auto-classifies target variables into **Regression** or **Classification** tasks.
* **Robust Preprocessing Pipeline:** Automated handling of categorical encodings (`OneHotEncoder`) and continuous feature scaling (`StandardScaler`).
* **Live Custom Value Inference:** Train a model on your dataset, then enter custom field values in real-time to generate live predictions.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Data Processing & Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/)
* **Visualization:** [Plotly](https://plotly.com/), [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/)

---


## 📂 Project Structure

```text
data-analytics-project/
│
├── app.py                         # Main Streamlit application entry point & routing
├── requirements.txt               # Dependencies package file
├── .gitignore                     # Git exclusion rules
│
├── modules/
│   ├── advanced_analytics.py      # ML pipeline training & custom prediction UI
│   ├── auto_dashboard.py          # Dynamic summary visual dashboard
│   ├── data_cleaning.py           # Preprocessing & transformation engine
│   ├── dataset_health.py          # Data Quality Index (DQI) & audit checks
│   ├── statistical_analysis.py    # Correlation matrices & distribution tests
│   ├── type_detection.py         # Pattern-based semantic data type detection
│   ├── version_history.py         # Undo/redo state tracking
│   ├── visualizations.py          # Custom Plotly chart templates
│   └── utils.py                   # Caching functions & helper methods
│
└── archive/                       # Legacy UI components






🚀 Quickstart & Installation
1. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/data-analytics-ml-suite.git](https://github.com/YOUR_USERNAME/data-analytics-ml-suite.git)
cd data-analytics-ml-suite

2. Set Up a Virtual Environment (Recommended)
On Windows:

python -m venv venv
venv\Scripts\activate

On macOS / Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Launch the Streamlit App

streamlit run app.py


---


🎯 How to Use
Upload Dataset: Load your CSV or Excel file through the sidebar interface.

Review Data Health: Navigate to Overview & Health to view missing value ratios and data quality scores.

Clean Data: Use Data Cleaning to handle null values, remove duplicates, or adjust column types.

Explore Trends: Analyze correlations, feature distributions, and chart recommendations under Exploratory Analytics.

Predict & Test: Open Predictive Modeling, choose your target variable, train a model, and fill out the live input form to test predictions on new data.


---
📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

MIT License

Copyright (c) 2026 G. N. Manoj Balaji

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

