# ABC Retain Suite – Valuable ChurnRadar  
Customer Segmentation and Valuable Churn Project (Superstore)

---

## 1. Executive summary

This repository contains a complete **Machine Learning project applied to relationship marketing**, using a *Superstore / Marketing Campaign* type dataset.

The main objective is twofold:

1. **To understand the behavior of the customer base** through:
   - EDA,
   - CDA (statistical analysis),
   - K-Means segmentation.

2. **To classify customers at risk of “valuable churn”**  
   (high-value customers who have stopped buying) and expose it in an app:

> **ABC Retain Suite – Module 1: Valuable ChurnRadar**  
> App + update service to help prioritize retention campaigns.

---

## 2. Project objectives

### Business objective

- Identify **which valuable customers** are stopping their purchases.
- Deliver customer **segments** with business meaning.
- Generate a **prioritized list for retention campaigns**:
  - who to call,
  - who to email,
  - which segment to focus on first.

### Data Science objective

- Build a complete pipeline:

  1. **0 – Master split**: separate train/test before everything else.
  2. **1 – EDA**: understand the dataset and create variables.
  3. **2 – CDA**: validate statistically what was observed in the EDA.
  4. **3 – K-Means segmentation**: 4 behavioral clusters.
  5. **4 – Supervised valuable churn model**.
  6. **5 – Final evaluation on test_master**.

- Integrate the model into a **Streamlit** app that performs scoring and allows campaign export.

---

## 3. Technology stack

- **Python 3.x**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **Joblib**
- **Streamlit**

---

## 4. Repository structure

```text
Proyecto_ML/
├── data/
│   ├── superstore_data.csv           # Original dataset (raw)
│   ├── superstore_modelado.csv       # Modeled dataset
│   ├── superstore_para_retencion     # Dataset for retention
│   ├── superstore_master.csv         # Clean dataset + features (before the split)
│   ├── train_master.csv              # Training split
│   ├── test_master.csv               # Final test split (untouched until the end)
│ │
├── notebooks/
│   ├── 0_Split_Master_Superstore.ipynb   # Purist train/test split
│   ├── 1_EDA_Superstore.ipynb            # EDA + variable creation
│   ├── 2_CDA_Superstore.ipynb            # CDA + valuable churn definition
│   ├── 3_Segmentacion_Clientes.ipynb     # K-Means + cluster interpretation
│   ├── 4_Modelo_Churn_Valioso.ipynb      # Supervised model + validation
│   └── 5_Evaluacion_TestMaster.ipynb     # Final evaluation on test_master
│
├── utils/
│   └── data_overview.py              # Class/functions for quick data overview
│
├── models/
│   └── churn_pipeline.pkl            # Preprocessing pipeline + trained model
│
├── app/
│   └── ABC_Retain_Suite.py   # Streamlit app (module 1 of the suite)
│
├── reports/
│   └── ABC_Retai_Suite_Tecnico.pdf # Technical project presentation
│   └── ABC_Retai_Suite.pdf         # Business presentation
│
└── README.md                         # This document
```

*(The names of some files may vary slightly, but this is the intended structure.)*

---

## 5. Data and key variables

### 5.1 Dataset

- Customer base with information on:
  - Registration date,
  - Purchases through different channels,
  - Total amount spent,
  - Website visits,
  - Basic demographic variables (Income, Kidhome, Teenhome, Education, Marital_Status).

### 5.2 Main feature engineering

In the EDA, key variables for customer behavior are created:

- **Recency**: days since the last purchase.
- **CustomerTenure**: customer tenure (days since registration).
- **MntTotal**: total historical spend.
- **TotalPurchases**: total number of purchases.
- **Perc_WebPurchases**: % of purchases made online.
- **Perc_CatalogPurchases**: % of purchases made through catalogs.
- **Perc_StorePurchases**: % of purchases made in physical stores.
- **NumWebVisitsMonth**: monthly website visits.
- **CLV_simple** = `MntTotal * TotalPurchases`.
- **CLV_log** = `log1p(MntTotal) * log1p(TotalPurchases)`  
  (more stable, used to define customer value and in the CDA,  
  **it is not used as a feature in the supervised model**).

### 5.3 Valuable Churn K-Means definition

1. **K-Means is applied to Recency** to find the most inactive cluster.
2. The most inactive cluster has an average of ≈ **83 days without purchasing**.
3. The conditions are defined as:
   - **Inactive**: `Recency ≥ 83`.
   - **Valuable**: `CLV_log ≥ median`.
4. Labels:
   - `Churn_Valioso_KMeans = 1` if (Inactive & Valuable), otherwise 0.
   - `Churn_KMeans = 1` if `Recency ≥ 83` (simple churn, inactivity only).

Overall results (in the original dataset):

- **Valuable Churn K-Means** ≈ **8.1 %**.
- **Simple churn (K-Means)** ≈ **16.5 %**.

The CDA shows that **Valuable Churn** is much more informative than simple churn.

---

## 6. Workflow and notebooks

### 6.1 `0_Split_Master_Superstore.ipynb`

**Objective:**  
Perform the **purist** train/test split before any analysis, in order to reserve a completely untouched **test_master** for the final model evaluation.

**Main steps:**

1. Load `superstore_master.csv` (dataset already cleaned and with basic features).
2. Perform a stratified split on the valuable churn label (`Churn_Valioso_KMeans`).
3. Save:
   - `train_master.csv`
   - `test_master.csv`  
     in the `data/` folder.

---

### 6.2 `1_EDA_Superstore.ipynb`

**Objective:**  
Initial exploration of the training data and variable creation.

**Steps:**

1. Load `train_master.csv`.
2. Quick review of:
   - sizes,
   - variable types,
   - missing values,
   - descriptive statistics.
3. Create behavioral variables (if they were not already created).
4. Visualizations:
   - distributions of Recency, MntTotal, TotalPurchases, Income, etc.
   - channel mix,
   - website visits.
5. Initial exploration of simple churn and valuable churn (without formal CDA yet).

---

### 6.3 `2_CDA_Superstore.ipynb`

**Objective:**  
Statistically validate what was observed in the EDA and **formally define valuable churn**.

**Steps:**

1. Confirm the definition of `Churn_Valioso_KMeans`:
   - K-Means on Recency → inactivity threshold ≈ 83 days.
   - Combination with CLV_log ≥ median.
2. Calculate rates:
   - Valuable churn ≈ 8.1 %,
   - Simple churn ≈ 16.5 %.
3. **Numerical CDA (Mann-Whitney)**:
   - Variables analyzed: Recency, MntTotal, TotalPurchases, Income, Perc_CatalogPurchases, NumWebVisitsMonth, Perc_StorePurchases, etc.
   - p-values and **rank-biserial** are reported.
   - The conclusion is that:
     - Recency has a very large effect (≈ 0.91),
     - MntTotal, TotalPurchases, Income, Perc_CatalogPurchases are relevant,
     - NumWebVisitsMonth is inversely associated.
4. **Categorical CDA (Chi-square)**:
   - Kidhome, Teenhome show a significant association with valuable churn.
   - Education, Marital_Status have a milder effect.
5. Conclusions:
   - Valuable churn **is not random**.
   - There is enough signal to build a supervised model.

---

### 6.4 `3_Segmentacion_Clientes.ipynb`

**Objective:**  
Segment customers into **4 clusters** with K-Means and describe them in business language. Cross the segmentation with valuable churn.

**Features for clustering:**

- Recency, CustomerTenure, MntTotal, TotalPurchases, Income,
- Perc_WebPurchases, Perc_CatalogPurchases, Perc_StorePurchases,
- NumWebVisitsMonth,
- Kidhome, Teenhome.

**Steps:**

1. Simple imputation and **standard scaling** of numerical variables.
2. K-Means with `k = 4` (trade-off between simplicity and interpretability).
3. Calculation of averages by cluster → profile table.
4. Heatmaps and visualizations comparing clusters.
5. **Cross with Churn_Valioso_KMeans**:
   - Valuable churn percentages are calculated within each cluster.
   - High-risk vs low-risk segments are identified.
6. Business interpretation:
   - Segments with high value and high risk,
   - Low-value segments with almost no valuable churn,
   - Differences by channel and demographic profile.

---

### 6.5 `4_Modelo_Churn_Valioso.ipynb`

**Objective:**  
Train a supervised model to classify **valuable churn** using only `train_master.csv` (without touching test_master).

**Target:**

- `Churn_Valioso_KMeans`.

**Features used:**

- Numerical:
  - Recency, MntTotal, TotalPurchases, Income,
  - Perc_CatalogPurchases, NumWebVisitsMonth,
  - Kidhome, Teenhome.
- Categorical:
  - Education, Marital_Status.

> ⚠️ **CLV_log is NOT used as a feature**, even though it was used to define valuable churn.  
> This prevents information leakage.

**Preprocessing and model:**

- `ColumnTransformer`:
  - Numerical → `SimpleImputer(median)` + `StandardScaler`.
  - Categorical → `SimpleImputer(most_frequent)` + `OneHotEncoder`.
- `Pipeline` with:
  - `preprocess` + `model`.

**Models and tuning:**

- LogisticRegression:
  - Penalty: L1 / L2,
  - C: [0.1, 1.0, 10.0],
  - `class_weight='balanced'`.
- RandomForestClassifier:
  - `n_estimators`: [100, 300],
  - `max_depth`: [None, 5, 10],
  - `min_samples_split`: [2, 5],
  - `class_weight='balanced'`.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- `GridSearchCV` with **ROC-AUC** as the metric.

**Validation results (hold-out):**

- Best model: **RandomForestClassifier**  
  (`n_estimators=100`, `max_depth=5`, `min_samples_split=2`, `class_weight='balanced'`).
- Metrics:
  - ROC-AUC ≈ **0.9933**.
  - Accuracy ≈ **0.9955**.
  - Precision (class 1 – valuable churn) ≈ **0.9722**.
  - Recall (class 1) ≈ **0.9722**.
- Importances:
  - Recency dominates,
  - then MntTotal, TotalPurchases, Income, Perc_CatalogPurchases,
  - and to a lesser extent Kidhome, NumWebVisitsMonth, Teenhome, and some categories.

**Output:**

- The complete pipeline (preprocessing + model) is saved in:
  - `models/churn_pipeline.pkl`.

---

### 6.6 `5_Evaluacion_TestMaster.ipynb`

**Objective:**  
Perform the **final evaluation** of the model using `test_master.csv` (never-seen data).

**Steps:**

1. Load `test_master.csv`.
2. Load `churn_pipeline.pkl`.
3. Obtain:
   - Valuable churn probabilities.
   - Final predictions.
4. Calculate test metrics:
   - ROC-AUC,
   - confusion matrix,
   - precision, recall, F1 for the positive class.
5. Compare validation vs test results to verify that there is no serious overfitting.
6. Leave the necessary columns ready (`Churn_Valioso_Pred`, `Churn_Valioso_Prob`) for the app to use.

---

## 7. Valuable churn model: summary

- **Problem type**: binary classification (`Churn_Valioso_KMeans` = 1 / 0).
- **Distribution** (in train):
  - ≈ 8% positive class (valuable churn),
  - ≈ 92% negative class.
- **Approach**:
  - Pipeline with preprocessing separating numerical and categorical features.
  - Compared models: Logistic Regression vs RandomForest.
  - Selection via GridSearchCV and ROC-AUC.
- **Best model**:
  - RandomForest with moderate depth.
- **Use in production / demo**:
  - The model is serialized with Joblib,
  - The Streamlit app uses it to perform scoring.

---

## 8. K-Means segmentation: summary

- **Number of clusters**: k = 4.

- **Variables**:
  - Recency, CustomerTenure, MntTotal, TotalPurchases, Income,
  - Perc_WebPurchases, Perc_CatalogPurchases, Perc_StorePurchases,
  - NumWebVisitsMonth, Kidhome, Teenhome.

- **Preprocessing**:
  - Imputation + StandardScaler.

- **Interpretation**:
  - Profiles such as the following are obtained:
    - **High-spend, high-frequency** segments (premium customers).
    - **Low-spend, low-frequency** segments.
    - More digital segments vs more physical-store-oriented segments.
  - When crossed with valuable churn:
    - Some clusters concentrate a much higher % of valuable churn.
    - Others contribute almost no valuable churn → low investment priority.

---

## 9. App: ABC Retain Suite – Valuable ChurnRadar

The app is located at:

```text
app/app_ABC_Retain_Suite_ChurnRadar.py
```

### 9.1 Execution

From the project root folder:

```bash
cd Proyecto_ML
streamlit run app/app_ABC_Retain_Suite_ChurnRadar.py
```

Requirements:

- Have `streamlit` and the project dependencies installed.
- Have the trained model available in `models/churn_pipeline.pkl`.
- Have a dataset available with the same columns as `train_master.csv` / `test_master.csv`.

### 9.2 Tabs (according to the current design)

The app is conceived as **Module 1 of ABC Retain Suite**.

1. **📊 Executive dashboard**
   - Number of customers.
   - % of valuable churn (historical or predicted).
   - Total spend and average CLV.
   - Charts:
     - Distribution by cluster.
     - Valuable churn by cluster (%).

2. **🧩 Segmentation**
   - Size of each cluster.
   - Table with cluster averages.
   - Description of each segment in business language.
   - Filter to explore customers from a specific cluster.

3. **🔥 Valuable Churn**
   - Valuable churn distribution (0/1).
   - Distribution by risk level (High / Medium / Low).
   - If the dataset has a historical label, it shows:
     - confusion matrix,
     - classification_report.

4. **📤 Export campaigns**
   - Filters:
     - cluster,
     - risk level,
     - minimum churn probability,
     - top-N customers.
   - Table with prioritized customers.
   - Button to download a CSV ready for campaign activation.

---

## 10. How to reproduce the project

### 10.1 Requirements

1. Clone this repository.
2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

*(If there is no `requirements.txt`, install: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit.)*

### 10.2 Recommended execution order

1. Run `0_Split_Master_Superstore.ipynb`  
   → generates `train_master.csv` and `test_master.csv`.

2. Run `1_EDA_Superstore.ipynb`  
   → exploratory analysis on `train_master`.

3. Run `2_CDA_Superstore.ipynb`  
   → definition and validation of `Churn_Valioso_KMeans`.

4. Run `3_Segmentacion_Clientes.ipynb`  
   → K-Means and segment profiles.

5. Run `4_Modelo_Churn_Valioso.ipynb`  
   → model training and saving `churn_pipeline.pkl`.

6. Run `5_Evaluacion_TestMaster.ipynb`  
   → final evaluation on `test_master`.

7. Launch the Streamlit app:

```bash
streamlit run app/app_ABC_Retain_Suite_ChurnRadar.py
```

---

## 11. Limitations and future work

- The definition of **valuable churn** is based on:
  - Recency ≥ 83 days + CLV_log ≥ median.  
  The model reproduces that definition very well, but:
  - it is strongly anchored to Recency,
  - it works as an excellent **current-state classifier**,
  - it is not a pure *early warning* model (several months ahead).
- The dataset corresponds to a single business context:
  - generalization to other sectors requires retraining the model with their data.
- Future work:
  - redefine churn using time windows (30–60–90 days),
  - incorporate more temporal and digital signals,
  - build a specific **recommendation module (cross-sell / up-sell)**,
  - add advanced interpretability (SHAP, etc.) if the context requires it.

---

## 12. Contact / credits

This project is part of a **training and portfolio process in Data Science and Machine Learning **, and it is integrated as the first module of:

> **ABC Retain Suite** – tools to take care of the value of your customers.

```markdown
Author: Cindy Tatiana Marin Espinosa
Role: Data Scientist / Data-Driven Marketing Analyst
```
