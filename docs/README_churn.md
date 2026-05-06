# 🛒 E-Commerce Customer Churn Prediction

> End-to-end machine learning project predicting customer churn for an e-commerce platform.  
> **Tools:** Python · Scikit-learn Pipeline · GridSearchCV · SHAP · KNN Imputer · GitHub

---

## 📌 Project Overview

This project simulates a real analytics mission: identifying which customers are likely to stop purchasing — and why — so the business can act before it's too late.

**Business question:**
> *"Which customers are at risk of churning, and what actions can we take to retain them?"*

**Why churn matters:**
> Acquiring a new customer costs 5x more than retaining an existing one. Predicting churn early gives the business a window to intervene with targeted retention strategies.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | E-Commerce Customer Churn Dataset (Kaggle) |
| Rows | 5,630 customers |
| Features | 20 columns |
| Target | Churn (0 = Retained · 1 = Churned) |
| Class balance | 83% Retained · 17% Churned |

---

## 🔍 Key Insights

- **📅 Tenure is the #1 predictor** — new customers (low tenure) churn significantly more than long-term customers. The median tenure for churners is ~3 months vs ~10 months for retained customers
- **😡 Complaints drive churn** — customers who filed a complaint in the last month are far more likely to leave, making complaint resolution a critical retention lever
- **📦 Recency matters** — customers who haven't ordered recently (high DaySinceLastOrder) show strong churn signals, confirming the classic RFM behavior
- **💰 Cashback protects loyalty** — customers receiving lower cashback amounts churn more, suggesting that reward programs have a measurable retention effect
- **💍 Single customers churn more** — MaritalStatus_Single showed a meaningful churn signal, possibly reflecting less stable purchasing habits

---

## 🤖 ML Workflow

### 1. Data Quality
- **7 columns** had missing values (4.5–5.5% each)
- Strategy: **KNN Imputer (k=5)** — chosen over mean/median to avoid distortion from outliers, and over row deletion to preserve information
- Outliers kept — values like 127km warehouse distance or 61 months tenure are realistic edge cases

### 2. Feature Engineering
- **One-Hot Encoding** applied to 5 categorical variables (PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus)
- Result: 20 → 36 features after encoding

### 3. Modeling — Sklearn Pipeline

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
])
```

**Why Pipeline?** Prevents data leakage by applying transformations consistently across train/test splits and future production data.

**Why `class_weight='balanced'`?** The 83/17% imbalance is moderate — no need for synthetic data (SMOTE). The balanced weight adjusts the loss function internally without distorting the data distribution.

### 4. Evaluation

| Metric | Class 0 (Retained) | Class 1 (Churned) |
|---|---|---|
| Precision | 0.96 | 0.97 |
| Recall | 0.99 | 0.78 |
| F1-Score | 0.98 | 0.87 |
| **Accuracy** | **0.96** | |

### 5. Cross-Validation (5-fold)

| Fold | F1 Score |
|---|---|
| Fold 1 | 0.954 |
| Fold 2 | 0.891 |
| Fold 3 | 0.930 |
| Fold 4 | 0.930 |
| Fold 5 | 0.885 |
| **Mean** | **0.918** |
| **Std** | **0.026** |

Low standard deviation (0.026) confirms model stability — results are not due to a lucky split.

### 6. Hyperparameter Optimization — GridSearchCV

```python
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [5, 10, None]
}
```

45 fits (9 combinations × 5 folds). Best params: `n_estimators=100, max_depth=None` — confirming default configuration was already optimal.

---

## 🔬 SHAP Explainability

SHAP values reveal not just *what* matters, but *how* each feature impacts individual predictions.

**Top features by SHAP importance:**

| Rank | Feature | Direction |
|---|---|---|
| 1 | Tenure | Low tenure → churn ↑ |
| 2 | Complain | Complaint filed → churn ↑ |
| 3 | DaySinceLastOrder | Many days inactive → churn ↑ |
| 4 | CashbackAmount | Low cashback → churn ↑ |
| 5 | MaritalStatus_Single | Single → churn ↑ |

---

## 🛠️ Tools Used

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML & Preprocessing | Scikit-learn (Pipeline, GridSearchCV, KNNImputer, StandardScaler, RandomForest) |
| Explainability | SHAP |
| AI Assistance | Claude AI, GitHub Copilot |
| Version Control | Git, GitHub |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/FernandDevCode/ecommerce-churn.git
cd ecommerce-churn

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn shap

# 3. Run the notebook
jupyter notebook notebooks/01_exploration.ipynb
```

---

## 📁 Project Structure

```
ecommerce-churn/
│
├── data/
│   └── ecommerce_churn.xlsx        # Raw dataset
│
├── notebooks/
│   └── 01_exploration.ipynb        # Full EDA + Pipeline + SHAP
│
├── docs/
│   └── business_recommendations.docx  # Business report
│
└── README.md
```

---
## 📊 Live Dashboard
👉 [View on Looker Studio](https://datastudio.google.com/s/uNgU26vKRag)



## 👤 Author

**Fernand Kissira SOHOU**  
Computer Science & Telecommunications Student  
 · [GitHub](https://github.com/FernandDevCode) · fernandkissiras@gmail.com

---

*Built as part of a data analytics portfolio targeting the Canadian job market.*
