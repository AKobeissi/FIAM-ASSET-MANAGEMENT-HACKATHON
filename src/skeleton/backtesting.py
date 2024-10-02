import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import os
import warnings

warnings.filterwarnings('ignore')

# RollingTimeSeriesCV class to handle rolling splits for time series cross-validation
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

            print(f"Split {i}: Train {train_start_date} to {train_end_date}, Test {test_start_date} to {test_end_date}")

            yield train_start_date, train_end_date, test_start_date, test_end_date

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

# StockPredictionPipeline class for preparing data, training the model, and making predictions
class StockPredictionPipeline:
    def __init__(self, stock_vars, ret_var):
        self.stock_vars = stock_vars
        self.ret_var = ret_var
        self.scaler = StandardScaler()

    def prepare_data(self, data):
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        data['next_month_return'] = data.groupby('stock_ticker')[self.ret_var].shift(-1)
        return data.dropna()

    def train_model(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        model = Ridge()
        model.fit(X_scaled, y_train)
        return model

    def predict(self, model, X_test):
        X_scaled = self.scaler.transform(X_test)
        return model.predict(X_scaled)

# Rolling cross-validation and stock prediction function
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

        model = pipeline.train_model(X_train, y_train)
        y_pred = pipeline.predict(model, X_test)

        mse = mean_squared_error(y_test, y_pred)
        predictions = test_data[['date', 'stock_ticker']].copy()
        predictions['predicted_return'] = y_pred
        predictions['actual_return'] = y_test.values

        results.append({
            'train_period': (train_start_date, train_end_date),
            'test_period': (test_start_date, test_end_date),
            'mse': mse,
            'predictions': predictions
        })

    return results

# Function to select the top 100 stocks based on predicted returns for each test period
def create_portfolio(results, top_n=100):
    portfolios = []
    for result in results:
        predictions = result['predictions']
        top_stocks = predictions.nlargest(top_n, 'predicted_return')
        portfolios.append({
            'test_period': result['test_period'],
            'top_stocks': top_stocks[['stock_ticker', 'predicted_return']]
        })
    return portfolios

# Save the predictions and portfolio results to CSV files
def save_results(results, portfolios, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, result in enumerate(results):
        test_period = f"{result['test_period'][0].strftime('%Y-%m-%d')}_to_{result['test_period'][1].strftime('%Y-%m-%d')}"
        #test_period = f"{result['test_period'][0]}_to_{result['test_period'][1]}"
        predictions = result['predictions']

        # Save predictions for the test period
        predictions_file_path = os.path.join(output_dir, f"predictions_{test_period}.csv")
        predictions.to_csv(predictions_file_path, index=False)
        print(f"Saved predictions for {test_period} at {predictions_file_path}")

        # Save portfolio (top 100 stocks) for the test period
        portfolio = portfolios[i]['top_stocks']
        portfolio_file_path = os.path.join(output_dir, f"portfolio_{test_period}.csv")
        portfolio.to_csv(portfolio_file_path, index=False)
        print(f"Saved portfolio for {test_period} at {portfolio_file_path}")

if __name__ == "__main__":
    print(datetime.datetime.now())

    # turn off pandas Setting with Copy Warning
    pd.set_option("mode.chained_assignment", None)

    # set working directory
    work_dir = "C:/Users/akobe/OneDrive/Asset-Management-FIAM/McGill-FIAM Asset Management Hackathon/data/"

    # read sample data
    file_path = os.path.join(
        work_dir, "hackathon_sample_v2.csv"
    )  # replace with the correct file name
    raw = pd.read_csv(
        file_path, parse_dates=["date"], low_memory=False
    )  # the date is the first day of the return month (t+1)

    # read list of predictors for stocks
    file_path = os.path.join(
        work_dir, "factor_char_list.csv"
    )  # replace with the correct file name
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    # define the left hand side variable
    ret_var = "stock_exret" #possibly change?
    new_set = raw[
        raw[ret_var].notna()
    ].copy()  # create a copy of the data and make sure the left hand side is not missing

    # transform each variable in each month to the same scale
    monthly = new_set.groupby("date")
    data = pd.DataFrame()
    for date, monthly_raw in monthly:
        group = monthly_raw.copy()
        # rank transform each variable to [-1, 1]
        for var in stock_vars:
            var_median = group[var].median(skipna=True)
            group[var] = group[var].fillna(
                var_median
            )  # fill missing values with the cross-sectional median of each month

            group[var] = group[var].rank(method="dense") - 1
            group_max = group[var].max()
            if group_max > 0:
                group[var] = (group[var] / group_max) * 2 - 1
            else:
                group[var] = 0  # in case of all missing values
                print("Warning:", date, var, "set to zero.")

        # add the adjusted values
        data = data._append(
            group, ignore_index=True
        )  # append may not work with certain versions of pandas, use concat instead if needed

    # initialize the starting date, counter, and output data
    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()

    #data_path = r"C:\Users\akobe\OneDrive\Asset-Management-FIAM\McGill-FIAM Asset Management Hackathon\data\hackathon_sample_v2.csv"  # Replace with your actual data path
    #stock_vars = list(pd.read_csv(r"C:\Users\akobe\OneDrive\Asset-Management-FIAM\McGill-FIAM Asset Management Hackathon\data\factor_char_list.csv")["variable"])  # Replace with your stock variable names
    ret_var = 'stock_exret'  # Replace with your return variable

    #data = pd.read_csv(data_path)

    results = run_rolling_cross_validation(data, stock_vars, ret_var, train_duration=120, test_duration=12, lookahead=1, n_splits=12)

    # Step 2: Create a portfolio selecting the top 100 stocks based on predicted returns
    portfolios = create_portfolio(results, top_n=100)

    # Step 3: Save the predictions and portfolio results
    output_dir = "results"
    save_results(results, portfolios, output_dir)
