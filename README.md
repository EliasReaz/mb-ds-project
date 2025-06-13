
### Forecasting Ferry Ticket Redemptions & Sales - Applied Scientist Technical Test

**Opportunity No. 44081**

## Project Summary

This project improves upon an existing forecasting model for **ferry ticket redemptions** and develops a new model for **ticket sales** using Toronto Island Park data. The models account for seasonality, spikes, and business behavior. The approach emphasizes explainability, reproducibility, and sound development workflow.

---

## Repository Structure

```
.
├── data/                         # Raw ferry ticket data
├── plot_eda/                     # Saved visualizations from EDA and modeling
├── eda.ipynb                     # Initial data exploration
├── Modeling.ipynb                # Modeling redemption counts, complie Model.py first
├── SalesCountForcasting.ipynb   # Forecasting ticket sales
├── Model.py                      # Main forecasting class with Base, Prophet, XGBoost+Residuals
├── requirements.txt              # Python dependency list
├── environment.yml               # Conda environment file
├── .gitignore                    # Git version control rules
├── README.md                     # Project overview and instructions
├── Summary_non_technical.md      # Summary for non-technical audience
└── Technical_summary.md          # Detailed technical report and methodology
```

---

## Objectives

- Improve baseline redemption forecast
- Build a new ticket sales forecast
- Ensure business-aligned modeling assumptions
- Communicate results clearly to both technical and non-technical audiences

---

## Key Features

- **Baseline Model**: Seasonal decomposition
- **Prophet Model**: Captures weekly and yearly patterns using additive models
- **XGBoost with Residual Correction**: Addresses spikes and nonlinearities
- **Sales Forecasting**: Implemented separately in `SalesCountForcasting.ipynb`
- **Explainability**: SHAP values and gain-based feature importance
- **Diagnostics**: Q-Q plots, residual distribution, fold-wise error metrics

---

## Assumptions

- Focus is on post-2022 (post-COVID) ferry usage trends, as this period reflects more stable and consistent behavior, free from major disruptions or anomalies.
- Redemption is a function of lagged sales, calendar seasonality, and recent demand.
- Real-time sales data is not available for same-day forecasting (lagged features used).

---

## Installation & Setup

1. Clone this repository
2. Create environment:

```bash
conda env create -f environment.yml
conda activate mb-ds-project
```

Or use `pip`:

```bash
pip install -r requirements.txt
```

---

## Usage

`SalesCountForecasting.ipynb` for forecasting Sales Count is an independent Jupyter Notebook with comments, notes, and observations.

To run the modeling pipeline:
```python
from Model import RedemptionModel

model = RedemptionModel(X=df, target_col='Redemption Count')
model.run_models()
model.plot_residual_distribution("XGBoost+Residuals")
model.train_final_model_and_show_explainability()
```

---

## Disclosure

This submission was prepared independently. AI tools (e.g., ChatGPT) were used in this project primarily as an alternative to online searches, documentation lookup, and Stack Overflow. They helped clarify concepts, suggest explanations, and assist with documentation.

---

## 📜 License

This repository is submitted as part of a confidential assessment and is not intended for public distribution or reuse.

---
