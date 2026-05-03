# ABC Retain Suite – ChurnRadar 🔥
### Predict Valuable Churn Before It Happens

> **The problem:** Most retention strategies react too late — the customer is already gone. ChurnRadar uses machine learning to identify high-value customers showing early signs of disengagement, so retention teams can act *before* revenue walks out the door.

---

## What it does

ChurnRadar is an end-to-end ML product that analyzes customer behavior, identifies at-risk high-value accounts, and delivers an actionable, campaign-ready prioritization list — ranked by churn probability.

**In practice, this means:**
- Upload your customer base → the model scores every customer's churn risk
- Filter by segment, risk level, or minimum probability
- Export a prioritized CSV ready for your CRM or retention campaign
- Know *exactly* who to call, email, or offer a discount — and in what order

---

## Business Impact

| Metric | Result |
|---|---|
| Churn prediction accuracy | **ROC-AUC: 0.9933** on unseen data |
| High-value customers at risk identified | ~8% of base (the most expensive to lose) |
| Segmentation | 4 behavioral clusters with distinct retention priorities |
| Actionability | One-click campaign export, ranked by probability |

> Focusing retention spend on the top 20% highest-risk, highest-value customers can protect **80%+ of at-risk revenue**.

---

## The Model

**Problem type:** Binary classification — *will this high-value customer stop buying?*

**Target definition (Valuable Churn):**
A customer is flagged as valuable churn when:
1. They haven't purchased in ≥ 83 days (K-Means threshold on recency)
2. Their lifetime value is above the median (CLV)

This dual condition ensures we prioritize customers worth retaining — not just inactive accounts with low spend.

**Key signals the model uses:**
- `Recency` — days since last purchase (strongest signal)
- `Total spend` and `purchase frequency`
- `Customer lifetime value`
- `Channel mix` — catalog vs. web vs. in-store
- `Customer tenure` and demographics

**Pipeline:**
```
Raw data → Feature engineering → Preprocessing → RandomForest → Probability scores → Campaign list
```

**Model performance (holdout test set):**
- ROC-AUC: **0.9933**
- Precision (churn class): **0.9722**
- Recall (churn class): **0.9722**

---

## Customer Segments

ChurnRadar segments customers into 4 behavioral clusters before scoring:

| Segment | Profile | Churn Risk |
|---|---|---|
| High-value, high-frequency | Premium buyers, catalog-heavy | Medium — worth protecting |
| High-value, disengaged | Big spenders going quiet | **High — top priority** |
| Low-value, active | Regular small buyers | Low |
| Low-value, inactive | Occasional / lapsed | Low — low ROI to pursue |

> Retention budget should concentrate on Segment 2 — the most recoverable revenue.

---

## The App – ABC Retain Suite

Built with **Streamlit**, the app has 4 modules:

### 📊 Executive Panel
- Total customers, % at risk, total CLV
- Churn distribution by segment
- At-a-glance health of your customer base

### 🧩 Segmentation Explorer
- Behavioral profiles per cluster
- Average spend, recency, channel mix
- Filter and explore any segment

### 🔥 Churn Risk Dashboard
- Distribution: High / Medium / Low risk
- Confusion matrix and classification report (if historical labels available)
- Probability distribution across the base

### 📤 Campaign Export
- Filter by: segment, risk level, minimum probability, top-N customers
- Preview prioritized list
- **One-click CSV export** → ready to upload to HubSpot, Salesforce, or email tool

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/citmaes17/customer-churn-segmentation-ml
cd customer-churn-segmentation-ml

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order (see /notebooks)
# Then launch the app:
streamlit run app/app_ABC_Retain_Suite_ChurnRadar.py
```

**Notebook sequence:**
```
0_Split → 1_EDA → 2_CDA → 3_Segmentation → 4_Model → 5_Evaluation
```

---

## Tech Stack

`Python` · `scikit-learn` · `Pandas / NumPy` · `Streamlit` · `Matplotlib / Seaborn` · `Joblib`

---

## Roadmap

- [ ] Early warning model (30–60–90 day prediction windows)
- [ ] SHAP explainability layer (why is this customer at risk?)
- [ ] Cross-sell / upsell recommendation module
- [ ] API endpoint for real-time scoring
- [ ] Integration with CRM webhooks

---

## Author

**Cindy Tatiana Marín Espinosa**
Data Science & Marketing Analytics | The Bridge Bootcamp, Valencia 🇪🇸

*Building ML tools that make retention teams faster, smarter, and more focused.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/cindy-marine/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/citmaes17)


