
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
import seaborn as sns
import scipy.stats as stats
import shap

class RedemptionModel:
    
    def __init__(self, X, target_col):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {}

    def score(self, truth, preds):
        # return a dictionary with MAE, MSE, RMSE, R2
        return {
            'MAE': mean_absolute_error(truth, preds),
            'MSE': mean_squared_error(truth, preds),
            'RMSE': np.sqrt(mean_squared_error(truth, preds)),
            'R2': r2_score(truth, preds)
        }

    def run_models(self, n_splits=4, test_size=7):
        '''
        n_splits: int number of folds
        test_size: int forecasting window in days
        run models with Time-Series Splits
        '''
        # Split time-series at fixed intervals  
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0 # fold count
        for train_idx, test_idx in tscv.split(self.X):
            # Test/Train split by train index and test index
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]

            # Base model - Seasonal Decomposition
            preds = self._base_model(X_train, X_test)
            self._store_results('Base', cnt, X_test, preds)

            # Meta Prophet model
            preds = self._prophet_model(X_train, X_test)
            self._store_results('Prophet', cnt, X_test, preds)

            # XGBoost + Residual Correction (for spikes) model
            # We first fit a base XGBoost model to get initial predictions
            # and then fit a residual model to capture spikes.
            # Base XGBoost model
            # This will also prepare the features for the residual model.
            base_preds, residuals, X_test_prepared = self._xgb_model(X_train, X_test)
            self._store_results('XGBoost', cnt, X_test, base_preds)

            # Fit residual model for spike adjustment
            res_model = XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42)
            res_model.fit(X_test_prepared, residuals)
            spike_preds = res_model.predict(X_test_prepared)

            # Final adjusted predictions: add spike prediction
            final_preds = base_preds + spike_preds
            self._store_results('XGBoost+Residuals', cnt, X_test, final_preds)

            cnt += 1

    def _store_results(self, model_name, cnt, X_test, preds):
        # store results by model
        if model_name not in self.results:
            self.results[model_name] = {}
        # X_test[self.target_col]  # this is the ground truth
        self.results[model_name][cnt] = {'metrics': self.score(X_test[self.target_col], preds),
                                         "truth": X_test[self.target_col],
                                         "preds":preds}
        # plot comparing actual and predicted values
        self.plot(X_test[self.target_col], preds, model_name, cnt)
        # Quantile-Qauntile (Q-Q) plot: check residual normality
        # Check normality, skewness, outliers of residuals
        self.qq_plot(X_test[self.target_col] - preds, model_name, cnt)

    def _base_model(self, train, test):
        '''
        Our base, too-simple model.
        Your model needs to take the training and test datasets (dataframes)
        and output a prediction based on the test data.

        Please leave this method as-is.

        '''
        res = sm.tsa.seasonal_decompose(train[self.target_col], period=365)
        res_clip = res.seasonal.apply(lambda x: max(0, x))
        res_clip.index = res_clip.index.dayofyear
        res_clip = res_clip.groupby(res_clip.index).mean()
        res_dict = res_clip.to_dict()
        return pd.Series(index=test.index, data=[res_dict.get(doy, res_dict.get(365)) for doy in test.index.dayofyear])

    def _prophet_model(self, train, test):
        # Prepare training data for Prophet by renaming columns as expected by the API
        df = train.reset_index()[['Timestamp', self.target_col]].rename(columns={'Timestamp': 'ds', self.target_col: 'y'})
        # log transformation of y to tackle skewness of target values 
        df['y'] = np.log1p(df['y'])
        # Initialize Prophet model with both weekly and yearly seasonality
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        # Fit the model to the log-transformed data
        model.fit(df)
        # Prepare the future DataFrame with the dates from the test set
        future = test.reset_index()[['Timestamp']].rename(columns={'Timestamp': 'ds'})
        # get predictions in log scale
        forecast_log = model.predict(future)
        # Inverse log-transformation for predictions in original scale
        forecast = np.expm1(forecast_log['yhat'])
        # return as series
        return pd.Series(forecast.values, index=test.index)

    def _xgb_model(self, train, test):
        
        # Prepare the DataFrame for XGBoost
        df = pd.concat([train, test]).copy()
        # make the df concainating both train and test data
        df_concat = pd.concat([train, test], axis='index')
        # Ensure the index is a datetime index
        if not isinstance(df_concat.index, pd.DatetimeIndex):
            df_concat.index = pd.to_datetime(df_concat.index)
        # Print the shape and first few rows of the dataframe
        # print("Dataframe shape and head:")
        # print(df_concat.shape)
        # print("="*20)
        # print(df_concat.head())
        
        df = df_concat.copy()
        
        # day of week: 0=Monday, 1=Tuesday,..,5=Saturday, 6=Sunday
        df['dayofweek'] = df.index.dayofweek # capture
        # is_weekend: 1 if day of week is Saturday or Sunday, else 0
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        # month: 1=January, 2=February,..,12=December
        df['month'] = df.index.month
        # is_summer: 1 if month is June, July, or August, else 0
        df["is_summer"] = df.index.month.isin([6, 7, 8]).astype(int)
        
        # Lag features capture previous impacts on the target variable
        df['lag_1'] = df[self.target_col].shift(1)
        df['lag_7'] = df[self.target_col].shift(7)
        df['lag_14'] = df[self.target_col].shift(14)
        df['lag_28'] = df[self.target_col].shift(28)
        # Rolling features capture trends over time
        # Shift by 1 to avoid data leakage
        # and then calculate rolling means
        # Rolling means capture weekly trends over the last 1-week, 2-week, 4-week
        df['rolling_7'] = df[self.target_col].shift(1).rolling(7).mean()
        df['rolling_14'] = df[self.target_col].shift(1).rolling(14).mean()
        df['rolling_28'] = df[self.target_col].shift(1).rolling(28).mean()
        # Sales Count: We find that Sales Count is highly correlated with 
        # Redemption Count, so we use it as a feature
        # We've used Lag features for Sales Count, 
        # because we assume that previous days' Sales Count are available
        df['sales_lag_1'] = df['Sales Count'].shift(1)
        df['sales_lag_7'] = df['Sales Count'].shift(7)
        # Rolling means for Sales Count
        # Shift by 1 to avoid data leakage
        # and then calculate rolling means
        df['sales_rolling_7'] = df['Sales Count'].shift(1).rolling(7).mean()
        df['sales_rolling_14'] = df['Sales Count'].shift(1).rolling(14).mean()
        # drop rows with NaN values having lags and rolling means
        df.dropna(inplace=True)
        
        # Remove 'Sales Count' from the DataFrame if present, 
        # since we don’t know today’s sales in advance. 
        # Including it would lead to data leakage.
        if 'Sales Count' in df.columns:
            df.drop(columns=['Sales Count'], inplace=True)
        # Ensure train and test indices align with available lagged features
        # we dropped some beginning rows of df due to NaN values from lagging/rolling
        # some dates from train.index no longer exist in df.index
        train_df = df.loc[train.index.intersection(df.index)]
        test_df = df.loc[test.index.intersection(df.index)]

        X_train = train_df.drop(columns=[self.target_col])
        # Since Redemption Count distribution is rightly skewed, 
        # we apply log1p transformation (a)to reduce skewness
        # and (b) impact of outliers
        # log1p is equivalent to log(1 + x), which is useful for zero values
        y_train = np.log1p(train_df[self.target_col])
        
        X_test = test_df.drop(columns=[self.target_col])
        y_test = test_df[self.target_col]

        model = XGBRegressor(n_estimators=400, max_depth=3, subsample=0.7,
                             colsample_bytree=1.0, learning_rate=0.023,
                             reg_lambda=0.0, reg_alpha=10.0, random_state=42)
        # Fit the model
        model.fit(X_train, y_train)
        # Predict on the test set, we predict in log scale
        preds_log = model.predict(X_test)
        # Convert log-predictions back to original scale
        # np.expm1 is equivalent to exp(x) - 1, which is the inverse of log1p
        preds = np.expm1(preds_log)
        # Store residuals for later analysis
        # Residuals are the difference between actual and predicted values    
        residuals = y_test - preds
        # return predictions, residuals, X_test
        return pd.Series(preds, index=X_test.index), residuals, X_test

    def plot(self, truth, preds, label, cnt):
        '''
        plots the actual vs predicted values for a given model.
        truth: pd.Series, actual values
        preds: pd.Series, predicted values
        label: str, label for the model
        cnt: int, fold number for the model
        '''
        plt.figure(figsize=(15, 8))
        plt.plot(truth.index, truth, label='Actual', color='grey')
        plt.plot(preds.index, preds, label=f'{label} Prediction', color='red')
        plt.title(f'{label} Forecast - Fold {cnt}', fontsize=15)
        plt.xlabel('Date')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def qq_plot(self, residuals, model_name, cnt):
        '''
        Plots a Q-Q (Quantile-Quantile) plot to assess whether the residuals
        from the given model follow a normal distribution. This helps evaluate:
        - Model bias (if residuals deviate systematically,   
            consistently above/below the expected line).
        - Presence of outliers or heavy tails/skewness (non-normality).
        - Check whether residual corrections are needed
        '''
        plt.figure(figsize=(4, 3))
        # Q-Q plot: compare residuals to a standard normal distribution
        stats.probplot(residuals, dist="norm", plot=plt)
        # add title
        plt.title(f'Q-Q Plot of Residuals: {model_name} - Fold {cnt}')
        plt.tight_layout()
        plt.show()


    def plot_residual_distribution(self, model_name):
        '''
        Plots residual distributions for models across all folds. This helps
        - Check distribution of prediction errors or residuals - normal or skewed
        - See patterns, outliers, or bias in the model predictions
        - If residuals are not centered around zero, the model is biased
        - If most residuals are positive, model is underpredicting
        - If most residuals are negative, model is overpredicting
        - Comparing base model (e.g. XGBoost) & post-correction (XGBoost + residuals correction)
        '''
        # if model name exists 
        if model_name not in self.results:
            raise ValueError(f"No results found for model '{model_name}'")
        residuals = [] # store residuals from each fold
        dates = [] # track corresponding timestamps 
        # loop over each fold's results for specified model
        for cnt, result in self.results[model_name].items():
            preds = result.get('preds') # Model predictions
            truth = result.get('truth') # Ground truth/actual values
            if preds is not None and truth is not None:
                res = truth - preds
                residuals.extend(res.values) # extend flattens & add each ele individually
                dates.extend(res.index)
        if residuals:
            # print(len(truth.index), len(residuals))
            residuals = np.array(residuals)
            dates = np.array(dates)
            # residuals = residuals[~np.isnan(residuals)]
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
            # Scatter plot of residuals
            sns.scatterplot(x=dates, y=residuals, color='blue', alpha=0.7, ax=ax[0])
            ax[0].axhline(0, color='red', linestyle='--')
            ax[0].set_title(f'Residuals - {model_name}')
            ax[0].set_xlabel('Date')
            ax[0].set_ylabel('Residuals')
            ax[0].grid(True)
            # Histogram/Distribution of residuals
            sns.histplot(residuals, kde=True, ax=ax[1])
            ax[1].set_title(f'Residual Distribution - {model_name}')
            ax[1].set_xlabel('Residuals')
            ax[1].grid(True)
            plt.tight_layout()
            plt.show()

    def train_final_model_and_show_explainability(self):
        """
        Model explanability & insights, Feature Importance
        Train a final XGBoost model using the full dataset
        Computes and plots global feature importance (gain-based)
        Explain how each feature contributes to predictions
        Visualizes SHAP values for individual feature contributions
        Influential features driving the model decisions.
        Support model validation, transparency, and communication with stakeholders.
        """

        df = self.X.copy()
        df.index = pd.to_datetime(df.index)
        # day of week: 0=Monday, 1=Tuesday,..,5=Saturday, 6=Sunday
        df['dayofweek'] = df.index.dayofweek # capture
        # is_weekend: 1 if day of week is Saturday or Sunday, else 0
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        # month: 1=January, 2=February,..,12=December
        df['month'] = df.index.month
        # is_summer: 1 if month is June, July, or August, else 0
        df["is_summer"] = df.index.month.isin([6, 7, 8]).astype(int)
        # Lag features capture previous impacts on the target variable
        df['lag_1'] = df[self.target_col].shift(1)
        df['lag_7'] = df[self.target_col].shift(7)
        df['lag_14'] = df[self.target_col].shift(14)
        df['lag_28'] = df[self.target_col].shift(28)
        # Rolling features capture trends over time
        # Shift by 1 to avoid data leakage
        # and then calculate rolling means
        # Rolling means capture weekly trends over the last 1-week, 2-week, 4-week
        df['rolling_7'] = df[self.target_col].shift(1).rolling(7).mean()
        df['rolling_14'] = df[self.target_col].shift(1).rolling(14).mean()
        df['rolling_28'] = df[self.target_col].shift(1).rolling(28).mean()
        # Sales Count: We find that Sales Count is highly correlated with 
        # Redemption Count, so we use it as a feature
        # We've used Lag features for Sales Count, 
        # because we assume that previous days' Sales Count are available
        df['sales_lag_1'] = df['Sales Count'].shift(1)
        df['sales_lag_7'] = df['Sales Count'].shift(7)
        # Rolling means for Sales Count
        # Shift by 1 to avoid data leakage
        # and then calculate rolling means
        df['sales_rolling_7'] = df['Sales Count'].shift(1).rolling(7).mean()
        df['sales_rolling_14'] = df['Sales Count'].shift(1).rolling(14).mean()
        df.dropna(inplace=True)
        if 'Sales Count' in df.columns:
            df.drop(columns=['Sales Count'], inplace=True)
            
        X_full = df.drop(columns=[self.target_col])
        y_full = np.log1p(df[self.target_col])

        model = XGBRegressor(n_estimators=400, max_depth=3, subsample=0.7,
                             colsample_bytree=1.0, learning_rate=0.023,
                             reg_lambda=0.0, reg_alpha=10.0, random_state=42)
        model.fit(X_full, y_full)

        # Feature Importance
        importances = model.feature_importances_
        feat_names = X_full.columns
        sorted_idx = np.argsort(importances)
        plt.figure(figsize=(10, 6))
        plt.barh(feat_names[sorted_idx], importances[sorted_idx])
        plt.title('Feature Importance (Final Model)')
        plt.tight_layout()
        plt.show()

        # SHAP
        shap.initjs()
        explainer = shap.Explainer(model, X_full)
        shap_values = explainer(X_full)
        shap.summary_plot(shap_values, X_full)
