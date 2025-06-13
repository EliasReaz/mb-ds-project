
### Technical Summary

The objective of this project was to improve the accuracy of daily ferry redemption forecasts and develop a second model to predict ticket sales using data from the City of Toronto Open Data Portal.

#### Data Selection and Preprocessing
The dataset spans several years and includes structural breaks - specifically in 2017 (flood-related disruptions) and early 2020 (COVID-19 pandemic).
To ensure consistent seasonal patterns, I filtered the data and used **post-COVID data from 2022 onward**, where trends and seasonality stabilized.

#### Baseline Model Challenges
The original model used a simple seasonal decomposition approach. It failed to capture the complex dynamics of the time series, which included:
- **Multiple seasonalities**: weekly (weekend effect) and monthly (summer peak).
- **Sudden surges**: unexpected spikes in ferry usage due to holidays or events.

#### Model Exploration
I also experimented with SARIMA separately. Although it handled single-seasonal trends decently, it was not included in the final folder as it lacked the flexibility to model both seasonal cycles present in the data.
Three modeling approaches were explored:
- **SARIMA**: Good at handling one type of seasonality, but limited for datasets with both weekly and monthly cycles.
- **Prophet (by Meta)**: Captures multiple seasonalities and trends well, but underperformed on sudden spikes or residual noise.
- **XGBoost**: A gradient-boosted tree model that accepts manually engineered features and is more adaptable to irregular patterns and outliers.

#### Final Modeling Strategy
A two-phase approach using **XGBoost with residual correction** was adopted:
1. **Base Model (XGBoost)**: Learned regular behaviors using features like day-of-week, month, lagged redemption counts, rolling means, and lagged sales data. Care was taken to avoid data leakage (e.g., current-day values were not used).
2. **Residual Adjustment**: The residuals (errors) from the base model were modeled using another XGBoost model to correct for spikes and anomalies.

This approach improved accuracy significantly and produced more stable, unbiased forecasts. Residual diagnostics and Q-Q plots confirmed reduced skewness and better model calibration after correction.

#### Ticket Sales Forecasting
A second model was built for predicting **ticket sales** using the same framework. This supports upstream planning and complements redemption forecasting. The features and pipeline structure mirrored those used in the redemption model.

#### Explainability
To ensure transparency:
- **Feature importance plots** showed that recent redemption history were the strongest predictors.
- **SHAP values** provided insight into how individual features impacted predictions - helping validate that the model learned sensible patterns (e.g., increased activity on weekends and in summer).

#### Reproducibility and Workflow
- The codebase includes modular scripts and Jupyter notebooks for EDA, modeling, tuning, and evaluation.
- All experiments use open-source tools (`XGBoost`, `Prophet`, `statsmodels`, `SHAP`).
- The environment is reproducible via `environment.yml` or `requirements.txt`.

#### Summary
The final models:
- Better captured seasonal behavior and event-driven spikes.
- Were interpretable and reproducible.
- Outperformed the baseline on key metrics.

This solution balances **accuracy**, **transparency**, and **practical deployment readiness** for forecasting ferry usage and ticket sales.

