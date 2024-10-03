import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from pandas.tseries.offsets import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import os
import warnings

warnings.filterwarnings('ignore')

class RollingTimeSeriesCV:
    def __init__(self, train_duration, test_duration, lookahead, n_splits):
        self.test_duration = test_duration
        self.train_duration = train_duration
        self.lookahead = lookahead
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        unique_dates = X['date'].unique()
        days = sorted(unique_dates, reverse=True)

        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_duration
            test_start_idx = test_end_idx + self.test_duration - 1

            train_end_idx = test_start_idx + 1 + self.lookahead
            train_start_idx = train_end_idx + self.train_duration - 1

            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

        for split in split_idx:
            train_start_idx, train_end_idx, test_start_idx, test_end_idx = split

            train_start_date = days[train_start_idx] if train_start_idx < len(days) else None
            train_end_date = days[train_end_idx] if train_end_idx < len(days) else None
            test_start_date = days[test_start_idx] if test_start_idx < len(days) else None
            test_end_date = days[test_end_idx] if test_end_idx < len(days) else None

            yield train_start_date, train_end_date, test_start_date, test_end_date

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

def mean_decrease_impurity(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return pd.Series(importances[indices], index=feature_names[indices])

def mean_decrease_accuracy(model, X, y, cv):
    feature_importance = np.zeros(X.shape[1])
    for train, test in cv.split(X):
        model.fit(X.iloc[train], y.iloc[train])
        score_full = model.score(X.iloc[test], y.iloc[test])
        for i in range(X.shape[1]):
            X_t = X.copy()
            np.random.shuffle(X_t.iloc[:, i].values)
            score_permuted = model.score(X_t.iloc[test], y.iloc[test])
            feature_importance[i] += score_full - score_permuted
    return pd.Series(feature_importance / cv.n_splits, index=X.columns).sort_values(ascending=False)

def single_feature_importance(model, X, y, cv):
    feature_importance = np.zeros(X.shape[1])
    for train, test in cv.split(X):
        for i in range(X.shape[1]):
            model.fit(X.iloc[train, [i]], y.iloc[train])
            feature_importance[i] += model.score(X.iloc[test, [i]], y.iloc[test])
    return pd.Series(feature_importance / cv.n_splits, index=X.columns).sort_values(ascending=False)

class StockPredictionPipeline:
    def __init__(self, stock_vars, ret_var):
        self.stock_vars = stock_vars
        self.ret_var = ret_var

    def prepare_data(self, data):
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        data['next_month_return'] = data.groupby('stock_ticker')[self.ret_var].shift(-1)
        return data.dropna()

    def preprocess_features(self, X):
        X = X.copy()
        for var in self.stock_vars:
            var_median = X[var].median(skipna=True)
            X[var] = X[var].fillna(var_median)
            X[var] = X[var].rank(method="dense") - 1
            group_max = X[var].max()
            if group_max > 0:
                X[var] = (X[var] / group_max) * 2 - 1
            else:
                X[var] = 0
            print("done processing")
        return X
    
    #NEW
    def select_features(self, X, y, n_features=10):
        """
        Select features using a combination of methods:
        1. Mutual Information
        2. Random Forest Feature Importance
        3. Spearman Correlation
        
        Parameters:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        n_features (int): Number of features to select
        
        Returns:
        list: Names of selected features
        """
        X_processed = self.preprocess_features(X)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_processed), columns=X_processed.columns)
        print("start feature selectiom")
        # 1. Mutual Information
        mi_scores = mutual_info_regression(X_scaled, y)
        mi_scores = pd.Series(mi_scores, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        print("end MI")
        print(mi_scores)
        # 2. Random Forest Feature Importance
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_scaled, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
        rf_importance = rf_importance.sort_values(ascending=False)
        print("end rf")
        print(rf_importance)

        # 3. Spearman Correlation
        corr_scores = []
        for column in X_scaled.columns:
            corr, _ = spearmanr(X_scaled[column], y)
            corr_scores.append(abs(corr))
        corr_scores = pd.Series(corr_scores, index=X.columns)
        corr_scores = corr_scores.sort_values(ascending=False)
        
        print("ecorr_scores")
        print(corr_scores)
        # Combine scores
        combined_scores = mi_scores.rank() + rf_importance.rank() + corr_scores.rank()
        combined_scores = combined_scores.sort_values()

        print(combined_scores)
        # Select top features
        self.selected_features = combined_scores.head(n_features).index.tolist()

        print(self.select_features
              )
        return self.selected_features

    #NEW
    def plot_feature_importance(self, X, y):
        if self.selected_features is None:
            raise ValueError("Feature selection has not been performed. Call select_features first.")

        importance_df = pd.DataFrame({
            'Feature': self.selected_features,
            'Mutual Information': mutual_info_regression(X[self.selected_features], y),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42).fit(X[self.selected_features], y).feature_importances_,
            'Spearman Correlation': [abs(spearmanr(X[feat], y)[0]) for feat in self.selected_features]
        })
        
        importance_df = importance_df.sort_values('Random Forest', ascending=False)

        fig, axes = plt.subplots(3, 1, figsize=(12, 20))
        for i, method in enumerate(['Mutual Information', 'Random Forest', 'Spearman Correlation']):
            importance_df.plot(x='Feature', y=method, kind='bar', ax=axes[i])
            axes[i].set_title(f'{method} Feature Importance')
            axes[i].set_xticklabels(importance_df['Feature'], rotation=90)

        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png')
        plt.close()
    #NEW
    def compute_feature_importance(self, X, y, method='all', n_splits=10):
        cv = KFold(n_splits=n_splits)
        
        model = RandomForestRegressor(n_estimators=100, max_features=1, random_state=42)
        
        importances = {}
        
        if method in ['MDI', 'all']:
            model.fit(X, y)
            importances['MDI'] = mean_decrease_impurity(model, X.columns)
        
        if method in ['MDA', 'all']:
            importances['MDA'] = mean_decrease_accuracy(model, X, y, cv)
        
        if method in ['SFI', 'all']:
            importances['SFI'] = single_feature_importance(model, X, y, cv)
    
        return importances
    #NEW
    def plot_feature_importances(self, importances):
        for method, importance in importances.items():
            plt.figure(figsize=(10, 6))
            importance.plot(kind='bar')
            plt.title(f'{method} Feature Importance')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{method}.png')
            plt.close()
    #NEW

    def plot_feature_importances(self, importances):
        for method, importance in importances.items():
            plt.figure(figsize=(10, 6))
            importance.plot(kind='bar')
            plt.title(f'{method} Feature Importance')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{method}.png')
            plt.close()

    def train_model(self, X_train, y_train, model_type='ridge'):
        X_processed = self.preprocess_features(X_train)
        if model_type == 'ols':
            model = LinearRegression(fit_intercept=False)
        elif model_type == 'lasso':
            model = Lasso(fit_intercept=False)
        elif model_type == 'ridge':
            model = Ridge(fit_intercept=False)
        elif model_type == 'elastic_net':
            model = ElasticNet(fit_intercept=False)
        else:
            raise ValueError("Invalid model type")
        
        model.fit(X_processed[self.stock_vars], y_train)
        return model

    def predict(self, model, X_test):
        X_processed = self.preprocess_features(X_test)
        return model.predict(X_processed[self.stock_vars])

# Make sure to update the run_rolling_cross_validation function as well:
def run_rolling_cross_validation(data, stock_vars, ret_var, train_duration=120, test_duration=12, lookahead=1, n_splits=12):
    pipeline = StockPredictionPipeline(stock_vars, ret_var)
    data = pipeline.prepare_data(data)

    cv = RollingTimeSeriesCV(train_duration, test_duration, lookahead, n_splits)

    results = []
    for train_start_date, train_end_date, test_start_date, test_end_date in cv.split(data):
        train_data = data[(data['date'] >= train_start_date) & (data['date'] <= train_end_date)]
        test_data = data[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]

        X_train = train_data[stock_vars]
        y_train = train_data['next_month_return']

        X_test = test_data[stock_vars]
        y_test = test_data['next_month_return']

        models = {}
        predictions = {}
        for model_type in ['ols', 'lasso', 'ridge', 'elastic_net']:
            models[model_type] = pipeline.train_model(X_train, y_train, model_type)
            predictions[model_type] = pipeline.predict(models[model_type], X_test)

        result = {
            'train_period': (train_start_date, train_end_date),
            'test_period': (test_start_date, test_end_date),
            'predictions': test_data[['date', 'stock_ticker']].copy(),
            'actual_return': y_test.values
        }

        for model_type in predictions:
            result['predictions'][f'predicted_return_{model_type}'] = predictions[model_type]
            result[f'mse_{model_type}'] = mean_squared_error(y_test, predictions[model_type])

        results.append(result)

    return results

def create_portfolio(results, n_portfolios=10):
    portfolios = []
    for result in results:
        predictions = result['predictions']
        predictions['actual_return'] = result['actual_return']  # Add this line to include actual returns
        for model_type in ['ols', 'lasso', 'ridge', 'elastic_net']:
            pred_col = f'predicted_return_{model_type}'
            predictions[f'rank_{model_type}'] = predictions[pred_col].rank(method='first')
            predictions[f'decile_{model_type}'] = pd.qcut(predictions[f'rank_{model_type}'], q=n_portfolios, labels=False)
        
        portfolios.append({
            'test_period': result['test_period'],
            'portfolio_allocations': predictions
        })
    return portfolios

def calculate_portfolio_returns(portfolios, n_portfolios=10):
    portfolio_returns = []
    for portfolio in portfolios:
        allocations = portfolio['portfolio_allocations']
        test_period = portfolio['test_period']
        
        for model_type in ['ols', 'lasso', 'ridge', 'elastic_net']:
            decile_col = f'decile_{model_type}'
            returns = []
            for i in range(n_portfolios):
                port_return = allocations[allocations[decile_col] == i]['actual_return'].mean()
                returns.append(port_return)
            
            long_short_return = returns[-1] - returns[0]
            
            portfolio_returns.append({
                'test_period': test_period,
                'model': model_type,
                'portfolio_returns': returns,
                'long_short_return': long_short_return
            })
    
    return portfolio_returns


def calculate_performance_metrics(portfolio_returns):
    df_returns = pd.DataFrame(portfolio_returns)
    
    for model in ['ols', 'lasso', 'ridge', 'elastic_net']:
        model_returns = df_returns[df_returns['model'] == model]['long_short_return']
        
        sharpe_ratio = model_returns.mean() / model_returns.std() * np.sqrt(12)
        
        # CAPM Alpha calculation (assuming risk-free rate is 0 for simplicity)
        market_returns = df_returns[df_returns['model'] == model]['portfolio_returns'].apply(lambda x: x[-1])  # Using the highest decile as market proxy
        model_formula = "long_short_return ~ market_returns"
        model_fit = sm.ols(formula=model_formula, data=df_returns[df_returns['model'] == model]).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        alpha = model_fit.params['Intercept']
        t_statistic = model_fit.tvalues['Intercept']
        information_ratio = alpha / np.sqrt(model_fit.mse_resid) * np.sqrt(12)
        
        max_drawdown = (model_returns.cummax() - model_returns).max()
        
        print(f"Performance metrics for {model.upper()}:")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"CAPM Alpha: {alpha:.4f}")
        print(f"t-statistic: {t_statistic:.4f}")
        print(f"Information Ratio: {information_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown:.4f}")
        print("\n")

def save_results(results, portfolios, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, result in enumerate(results):
        test_period = f"{result['test_period'][0].strftime('%Y-%m-%d')}_to_{result['test_period'][1].strftime('%Y-%m-%d')}"
        predictions = result['predictions']

        predictions_file_path = os.path.join(output_dir, f"predictions_{test_period}.csv")
        predictions.to_csv(predictions_file_path, index=False)
        print(f"Saved predictions for {test_period} at {predictions_file_path}")

        portfolio = portfolios[i]['portfolio_allocations']
        portfolio_file_path = os.path.join(output_dir, f"portfolio_{test_period}.csv")
        portfolio.to_csv(portfolio_file_path, index=False)
        print(f"Saved portfolio for {test_period} at {portfolio_file_path}")

if __name__ == "__main__":
    print(datetime.datetime.now())

    pd.set_option("mode.chained_assignment", None)

    work_dir = "C:/Users/akobe/OneDrive/Asset-Management-FIAM/McGill-FIAM Asset Management Hackathon/data/"

    file_path = os.path.join(work_dir, "hackathon_sample_v2.csv")
    raw = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)

    stock_vars = list(pd.read_csv(os.path.join(work_dir, "factor_char_list.csv"))["variable"].values)

    ret_var = "stock_exret"
    
    pipeline = StockPredictionPipeline(stock_vars, ret_var)
    data = pipeline.prepare_data(raw)

    X = data[stock_vars]
    y = data['next_month_return']
    
    #NEW
    selected_features = pipeline.select_features(X, y, n_features=20)
    print(f"Selected features: {selected_features}")
    pipeline.plot_feature_importance(X, y)
    #NEW
    importances = pipeline.compute_feature_importance(X, y)
    pipeline.plot_feature_importances(importances)

    results = run_rolling_cross_validation(data, stock_vars, ret_var, train_duration=120, test_duration=12, lookahead=1, n_splits=12)

    portfolios = create_portfolio(results)

    portfolio_returns = calculate_portfolio_returns(portfolios)

    calculate_performance_metrics(portfolio_returns)

    output_dir = "results"
    save_results(results, portfolios, output_dir)

    print(datetime.datetime.now())
    
    
