import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
import statsmodels.api as sm 
from pandas.tseries.offsets import *
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import os
import warnings
import seaborn as sns
import riskfolio as rp

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

"""
class RLAgent:
    def __init__(self, n_portfolios, n_stocks, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_portfolios, n_stocks))  # Q-Table for learning state-action values
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.n_portfolios = n_portfolios  # Number of portfolio allocation actions
        self.n_stocks = n_stocks  # Number of stocks

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:  # Exploration
            return np.random.choice(range(self.n_portfolios))
        else:  # Exploitation
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def reset(self):
        self.q_table = np.zeros((self.n_portfolios, self.n_stocks))
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            
            with torch.no_grad():
                target = reward + self.gamma * torch.max(self.model(next_state)) * (1 - done)
            target_f = self.model(state)
            target_f[0][action] = target
            
            loss = self.criterion(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    return np.sqrt(12) * excess_returns.mean() / excess_returns.std()

def create_portfolio_with_dqn(results, n_portfolios=10, state_size=4, window_size=12):
    agent = DQNAgent(state_size, n_portfolios)
    portfolios = []
    all_returns = pd.DataFrame()
    
    for i, result in enumerate(results):
        predictions = result['predictions']
        predictions['actual_return'] = result['actual_return']

        # Calculate market indicators (state)
        market_return = predictions['actual_return'].mean()
        market_volatility = predictions['actual_return'].std()
        positive_sentiment = (predictions['actual_return'] > 0).mean()
        momentum = predictions['actual_return'].autocorr()

        state = np.array([market_return, market_volatility, positive_sentiment, momentum])

        action = agent.act(state)

        portfolio_returns = {}
        for model_type in ['ols', 'lasso', 'ridge', 'elastic_net']:
            pred_col = f'predicted_return_{model_type}'
            predictions[f'rank_{model_type}'] = predictions[pred_col].rank(method='first')
            predictions[f'decile_{model_type}'] = pd.qcut(predictions[f'rank_{model_type}'], q=n_portfolios, labels=False)

            portfolio_return = predictions.loc[predictions[f'decile_{model_type}'] == action, 'actual_return'].mean()
            portfolio_returns[model_type] = portfolio_return

        # Add returns to the dataframe
        all_returns = pd.concat([all_returns, pd.DataFrame([portfolio_returns])], ignore_index=True)

        # Calculate Sharpe ratio if we have enough data
        if len(all_returns) >= window_size:
            rolling_returns = all_returns.rolling(window=window_size)
            sharpe_ratios = rolling_returns.apply(calculate_sharpe_ratio)
            reward = sharpe_ratios.iloc[-1].mean()  # Average Sharpe ratio across models
        else:
            reward = 0  # No reward until we have enough data

        # Prepare next state
        next_market_return = all_returns.iloc[-1].mean()
        next_market_volatility = all_returns.iloc[-1].std()
        next_positive_sentiment = (all_returns.iloc[-1] > 0).mean()
        next_momentum = all_returns.iloc[-1].autocorr() if len(all_returns) > 1 else 0
        next_state = np.array([next_market_return, next_market_volatility, next_positive_sentiment, next_momentum])

        # Store the experience
        agent.remember(state, action, reward, next_state, False)

        # Train the agent
        if len(agent.memory) > 32:
            agent.replay(32)

        portfolios.append({
            'test_period': result['test_period'],
            'portfolio_allocations': predictions,
            'selected_portfolio': action,
            'portfolio_returns': portfolio_returns,
            'sharpe_ratio': reward
        })

    return portfolios, agent
"""
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            
            with torch.no_grad():
                target = reward + self.gamma * torch.max(self.model(next_state)) * (1 - done)
            target_f = self.model(state)
            target_f[0][action] = target
            
            loss = self.criterion(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def create_portfolio_with_dqn(results, n_portfolios=10, state_size=4):
    agent = DQNAgent(state_size, n_portfolios)
    portfolios = []
    
    for i, result in enumerate(results):
        predictions = result['predictions']
        predictions['actual_return'] = result['actual_return']

        # Calculate market indicators (state)
        market_return = predictions['actual_return'].mean()
        market_volatility = predictions['actual_return'].std()
        positive_sentiment = (predictions['actual_return'] > 0).mean()
        momentum = predictions['actual_return'].autocorr()

        state = np.array([market_return, market_volatility, positive_sentiment, momentum])

        portfolio_returns = {}
        for model_type in ['ols', 'lasso', 'ridge', 'elastic_net']:
            pred_col = f'predicted_return_{model_type}'
            predictions[f'rank_{model_type}'] = predictions[pred_col].rank(method='first')
            predictions[f'decile_{model_type}'] = pd.qcut(predictions[f'rank_{model_type}'], q=n_portfolios, labels=False)

            action = agent.act(state)

            portfolio_return = predictions.loc[predictions[f'decile_{model_type}'] == action, 'actual_return'].mean()
            portfolio_returns[model_type] = portfolio_return

        # Use the best performing model as the reward
        reward = max(portfolio_returns.values())

        # Prepare next state (you might want to improve this)
        next_market_return = reward  # Simplified for this example
        next_market_volatility = predictions['actual_return'].std()
        next_positive_sentiment = (predictions['actual_return'] > 0).mean()
        next_momentum = predictions['actual_return'].autocorr()
        next_state = np.array([next_market_return, next_market_volatility, next_positive_sentiment, next_momentum])

        # Store the experience
        agent.remember(state, action, reward, next_state, False)

        # Train the agent
        if len(agent.memory) > 32:
            agent.replay(32)

        portfolios.append({
            'test_period': result['test_period'],
            'portfolio_allocations': predictions,
            'selected_portfolio': action,
            'portfolio_returns': portfolio_returns
        })

    return portfolios, agent
"""
class RLAgent:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.n_actions = n_actions
        self.n_states = n_states

    def get_state(self, market_indicators):
        # Convert market indicators to a state
        state = 0
        for i, indicator in enumerate(market_indicators):
            state += indicator * (2 ** i)
        return int(state)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

def create_portfolio_with_rl(results, n_portfolios=10, rl_agent=None):
    if rl_agent is None:
        n_states = 2 ** 4  # 4 market indicators
        rl_agent = RLAgent(n_actions=n_portfolios, n_states=n_states)

    portfolios = []
    for i, result in enumerate(results):
        predictions = result['predictions']
        predictions['actual_return'] = result['actual_return']

        # Calculate market indicators
        market_return = predictions['actual_return'].mean()
        market_volatility = predictions['actual_return'].std()
        positive_sentiment = (predictions['actual_return'] > 0).mean()
        momentum = predictions['actual_return'].autocorr()

        # Convert indicators to binary for simplicity
        market_indicators = [
            int(market_return > 0),
            int(market_volatility > predictions['actual_return'].std().mean()),
            int(positive_sentiment > 0.5),
            int(momentum > 0)
        ]

        state = rl_agent.get_state(market_indicators)

        portfolio_returns = {}
        for model_type in ['ols', 'lasso', 'ridge', 'elastic_net']:
            pred_col = f'predicted_return_{model_type}'
            predictions[f'rank_{model_type}'] = predictions[pred_col].rank(method='first')
            predictions[f'decile_{model_type}'] = pd.qcut(predictions[f'rank_{model_type}'], q=n_portfolios, labels=False)

            action = rl_agent.choose_action(state)

            portfolio_return = predictions.loc[predictions[f'decile_{model_type}'] == action, 'actual_return'].mean()
            portfolio_returns[model_type] = portfolio_return

        # Use the best performing model as the reward
        reward = max(portfolio_returns.values())

        # Simulate next state (you might want to improve this)
        next_state = rl_agent.get_state([np.random.randint(0, 2) for _ in range(4)])

        rl_agent.update_q_table(state, action, reward, next_state)

        portfolios.append({
            'test_period': result['test_period'],
            'portfolio_allocations': predictions,
            'selected_portfolio': action,
            'portfolio_returns': portfolio_returns
        })

    return portfolios, rl_agent

def create_portfolio_with_hrp(results, n_stocks=50):
    portfolios = []
    
    for result in results:
        predictions = result['predictions']
        test_period = result['test_period']
        actual_returns = result['actual_return']

        combined_data = predictions.copy()
        combined_data['actual_return'] = actual_returns

        # Prepare the returns dataframe
        returns = combined_data.pivot(index='date', columns='stock_ticker', values='actual_return')
        returns = returns.dropna(axis=1)
        
        # Select n_stocks randomly (you can modify this selection criteria)
        selected_stocks = returns.sample(n=n_stocks, axis=1, random_state=42)
        
        # Create the HRP portfolio object
        port = rp.HCPortfolio(returns=selected_stocks)
        
        # Estimate optimal portfolio
        weights = port.optimization(model='HRP', 
                                    codependence='pearson', 
                                    rm='MV', 
                                    rf=0, 
                                    linkage='single', 
                                    max_k=10, 
                                    leaf_order=True)
        
        # Calculate portfolio return
        portfolio_return = (weights * returns.iloc[-1]).sum()
        
        portfolios.append({
            'test_period': test_period,
            'portfolio_allocations': weights,
            'portfolio_return': portfolio_return,
            'selected_stocks': selected_stocks

        })
    
    return portfolios

"""
def create_portfolio_with_rl(results, n_portfolios=10, n_stocks=100, rl_agent=None):
    if rl_agent is None:
        rl_agent = RLAgent(n_portfolios, n_stocks)

    portfolios = []
    for result in results:
        predictions = result['predictions']
        predictions['actual_return'] = result['actual_return']

        # RL agent selects portfolio allocation dynamically based on past returns
        for model_type in ['ols', 'lasso', 'ridge', 'elastic_net']:
            pred_col = f'predicted_return_{model_type}'
            predictions[f'rank_{model_type}'] = predictions[pred_col].rank(method='first')
            predictions[f'decile_{model_type}'] = pd.qcut(predictions[f'rank_{model_type}'], q=n_portfolios, labels=False)

            state = int(predictions[f'decile_{model_type}'].mean())  # Current market condition (mean decile)
            action = rl_agent.choose_action(state)  # Select action based on current state

            # Reward = actual return of selected portfolio
            portfolio_return = predictions.loc[predictions[f'decile_{model_type}'] == action, 'actual_return'].mean()
            reward = portfolio_return

            # Simulate next state as mean decile of the next prediction
            next_state = state  # Simplified for now (can be updated based on future predictions)
            rl_agent.update_q_table(state, action, reward, next_state)  # Update Q-values

            portfolios.append({
                'test_period': result['test_period'],
                'portfolio_allocations': predictions,
                'selected_portfolio': action,
                'portfolio_return': portfolio_return
            })

    return portfolios, rl_agent
"""
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

def select_top_stocks(predictions, model_type='ridge', n_stocks=100):
    # Sort stocks by predicted return in descending order
    sorted_stocks = predictions.sort_values(f'predicted_return_{model_type}', ascending=False)
    
    # Select the top N stocks
    top_stocks = sorted_stocks.head(n_stocks)
    
    return top_stocks

def create_portfolio(results, n_portfolios=10, n_top_stocks=100):
    portfolios = []
    for result in results:
        predictions = result['predictions']
        predictions['actual_return'] = result['actual_return']
        
        portfolio_data = {}
        for model_type in ['ols', 'lasso', 'ridge', 'elastic_net']:
            # Select top stocks based on this model's predictions
            top_stocks = select_top_stocks(predictions, model_type, n_top_stocks)
            
            pred_col = f'predicted_return_{model_type}'
            top_stocks[f'rank_{model_type}'] = top_stocks[pred_col].rank(method='first')
            top_stocks[f'decile_{model_type}'] = pd.qcut(top_stocks[f'rank_{model_type}'], q=n_portfolios, labels=False)
            
            portfolio_data[model_type] = top_stocks
        
        portfolios.append({
            'test_period': result['test_period'],
            'portfolio_allocations': portfolio_data
        })
    return portfolios

def calculate_portfolio_returns(portfolios, n_portfolios=10):
    portfolio_returns = []
    for portfolio in portfolios:
        allocations = portfolio['portfolio_allocations']
        test_period = portfolio['test_period']

        for model_type, allocations in allocations.items():
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

def calculate_performance_metrics1(portfolio_returns, market_returns=None, risk_free_rate=0.02):
    df_returns = pd.DataFrame(portfolio_returns)
    
    for model in ['ols', 'lasso', 'ridge', 'elastic_net']:
        model_returns = df_returns[df_returns['model'] == model]['long_short_return']
        
        # Annualized return
        ann_return = (1 + model_returns.mean()) ** 12 - 1
        
        # Annualized standard deviation
        ann_std = model_returns.std() * np.sqrt(12)
        
        # Sharpe Ratio (annualized)
        sharpe_ratio = (ann_return - risk_free_rate) / ann_std
        
        # Maximum drawdown
        cum_returns = (1 + model_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Maximum one-month loss
        max_monthly_loss = model_returns.min()
        
        # CAPM Alpha and Information Ratio calculation
        if market_returns is not None:
            X = sm.add_constant(market_returns)
            model_fit = sm.OLS(model_returns, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
            alpha = model_fit.params['const']
            beta = model_fit.params[1]
            residuals = model_fit.resid
            
            # Annualized Alpha
            ann_alpha = (1 + alpha) ** 12 - 1
            
            # Information Ratio (annualized)
            information_ratio = (alpha / residuals.std()) * np.sqrt(12)
        else:
            ann_alpha = np.nan
            information_ratio = np.nan
        
        # Portfolio Turnover (assuming we have portfolio weights for each period)
        # This is a placeholder and needs to be implemented based on actual portfolio weights
        portfolio_turnover = np.nan
        
        print(f"Performance metrics for {model.upper()}:")
        print(f"Average Annualized Return: {ann_return:.4f}")
        print(f"Annualized Standard Deviation: {ann_std:.4f}")
        print(f"Annualized Alpha: {ann_alpha:.4f}")
        print(f"Sharpe Ratio (annualized): {sharpe_ratio:.4f}")
        print(f"Information Ratio (annualized): {information_ratio:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4f}")
        print(f"Maximum One-Month Loss: {max_monthly_loss:.4f}")
        print(f"Portfolio Turnover: {portfolio_turnover:.4f}")
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

        portfolio_dict = portfolios[i]['portfolio_allocations']
        for model_type, portfolio_df in portfolio_dict.items():
            portfolio_file_path = os.path.join(output_dir, f"portfolio_{model_type}_{test_period}.csv")
            portfolio_df.to_csv(portfolio_file_path, index=False)
            print(f"Saved {model_type} portfolio for {test_period} at {portfolio_file_path}")

#PLOTS FOR DIFFERENT STRATEGIES
def plot_cumulative_returns(portfolios, benchmark_returns=None):
    """
    Plot cumulative returns of RL strategy vs benchmark (if provided)
    """
    rl_returns = pd.DataFrame([p['portfolio_returns'] for p in portfolios])
    cumulative_returns = (1 + rl_returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column], label=f'RL {column.upper()}')
    
    if benchmark_returns is not None:
        benchmark_cum_returns = (1 + benchmark_returns).cumprod()
        plt.plot(benchmark_cum_returns.index, benchmark_cum_returns, label='Benchmark', linestyle='--')
    
    plt.title('Cumulative Returns: RL Strategy vs Benchmark')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_returns.png')
    plt.close()

def calculate_performance_metrics2(portfolio_returns, is_dqn=False, risk_free_rate=0.02):
    df_returns = pd.DataFrame(portfolio_returns)
    for model in ['ols', 'lasso', 'ridge', 'elastic_net']:
        model_returns = df_returns['portfolio_returns'].apply(lambda x: x[model])
        
        # Annualized return
        ann_return = (1 + model_returns.mean()) ** 12 - 1
        
        # Annualized standard deviation
        ann_std = model_returns.std() * np.sqrt(12)
        
        # Sharpe Ratio (annualized)
        sharpe_ratio = (ann_return - risk_free_rate) / ann_std
        
        # Maximum drawdown
        cum_returns = (1 + model_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Maximum one-month loss
        max_monthly_loss = model_returns.min()
        
        # Annualized Alpha and Information Ratio
        # For simplicity, we'll use the average of all model returns as the market return
        market_returns = df_returns['portfolio_returns'].apply(lambda x: np.mean(list(x.values())))
        X = sm.add_constant(market_returns)
        model_fit = sm.OLS(model_returns, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        alpha = model_fit.params['const']
        beta = model_fit.params[1]
        residuals = model_fit.resid
        
        # Annualized Alpha
        ann_alpha = (1 + alpha) ** 12 - 1
        
        # Information Ratio (annualized)
        information_ratio = (alpha / residuals.std()) * np.sqrt(12)
        
        # Portfolio Turnover (placeholder, needs actual implementation)
        portfolio_turnover = np.nan
        
        print(f"Performance metrics for {model.upper()}:")
        print(f"Average Annualized Return: {ann_return:.4f}")
        print(f"Annualized Standard Deviation: {ann_std:.4f}")
        print(f"Annualized Alpha: {ann_alpha:.4f}")
        print(f"Sharpe Ratio (annualized): {sharpe_ratio:.4f}")
        print(f"Information Ratio (annualized): {information_ratio:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4f}")
        print(f"Maximum One-Month Loss: {max_monthly_loss:.4f}")
        print(f"Portfolio Turnover: {portfolio_turnover:.4f}")
        print("\n")
        
from pandas.tseries.offsets import MonthBegin

def turnover_count(df):
    # count the number of stocks at the beginning of each month
    start_stocks = df[["permno", "date"]].copy()
    start_stocks = start_stocks.sort_values(by=["date", "permno"])
    start_count = start_stocks.groupby(["date"])["permno"].count().reset_index()

    end_stocks = df[["permno", "date"]].copy()
    end_stocks["date"] = end_stocks["date"] - MonthBegin(1)  # shift the date to the beginning of the next month
    end_stocks = end_stocks.sort_values(by=["date", "permno"])

    remain_stocks = start_stocks.merge(end_stocks, on=["date", "permno"], how="inner")
    remain_count = remain_stocks.groupby(["date"])["permno"].count().reset_index()  # count the number of stocks that remain in the next month
    remain_count = remain_count.rename(columns={"permno": "remain_count"})

    port_count = start_count.merge(remain_count, on=["date"], how="inner")
    port_count["turnover"] = (port_count["permno"] - port_count["remain_count"]) / port_count["permno"]  # calculate the turnover as the average of the percentage of stocks that are replaced each month
    return port_count["turnover"].mean()

def calculate_performance_metrics3(portfolios, is_dqn=True, risk_free_rate=0.02):
    df_returns = pd.DataFrame(portfolio_returns)
    for model in ['ols', 'lasso', 'ridge', 'elastic_net']:
        model_returns = df_returns['portfolio_returns'].apply(lambda x: x[model])
         
         # Annualized return
        ann_return = (1 + model_returns.mean()) ** 12 - 1
        
        # Annualized standard deviation
        ann_std = model_returns.std() * np.sqrt(12)
        
        # Sharpe Ratio (annualized)
        sharpe_ratio = (ann_return - risk_free_rate) / ann_std
        
        # Maximum drawdown
        cum_returns = (1 + model_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Maximum one-month loss
        max_monthly_loss = model_returns.min()
        
        # Annualized Alpha and Information Ratio
        # For simplicity, we'll use the average of all model returns as the market return
        market_returns = df_returns['portfolio_returns'].apply(lambda x: np.mean(list(x.values())))
        X = sm.add_constant(market_returns)
        model_fit = sm.OLS(model_returns, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        alpha = model_fit.params['const']
        beta = model_fit.params[1]
        residuals = model_fit.resid
        
        # Annualized Alpha
        ann_alpha = (1 + alpha) ** 12 - 1
        
        # Information Ratio (annualized)
        information_ratio = (alpha / residuals.std()) * np.sqrt(12)
        
       # Portfolio Turnover
        if is_dqn:
            # For DQN, we'll use the change in selected portfolio as a proxy for turnover
            portfolio_changes = [abs(portfolios[i]['selected_portfolio'] - portfolios[i-1]['selected_portfolio']) 
                                 for i in range(1, len(portfolios))]
            portfolio_turnover = sum(portfolio_changes) / len(portfolios)
        else:
            # For non-DQN, we'll use the turnover_count function
            # Note: This assumes that the portfolio_allocations contain 'permno' and 'date' columns
            long_positions = pd.concat([p['portfolio_allocations'][p['portfolio_allocations'][f'decile_{model}'] == 9] 
                                        for p in portfolios])
            short_positions = pd.concat([p['portfolio_allocations'][p['portfolio_allocations'][f'decile_{model}'] == 0] 
                                         for p in portfolios])
            long_turnover = turnover_count(long_positions)
            short_turnover = turnover_count(short_positions)
            portfolio_turnover = (long_turnover + short_turnover) / 2

        print(f"Performance metrics for {model.upper()}:")
        print(f"Average Annualized Return: {ann_return:.4f}")
        print(f"Annualized Standard Deviation: {ann_std:.4f}")
        print(f"Annualized Alpha: {ann_alpha:.4f}")
        print(f"Sharpe Ratio (annualized): {sharpe_ratio:.4f}")
        print(f"Information Ratio (annualized): {information_ratio:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4f}")
        print(f"Maximum One-Month Loss: {max_monthly_loss:.4f}")
        print(f"Portfolio Turnover: {portfolio_turnover:.4f}")
        print("\n")
       
    
        
        results[model] = {
            "Average Annualized Return": ann_return,
            "Annualized Standard Deviation": ann_std,
            "Annualized Alpha": ann_alpha,
            "Sharpe Ratio (annualized)": sharpe_ratio,
            "Information Ratio (annualized)": information_ratio,
            "Maximum Drawdown": max_drawdown,
            "Maximum One-Month Loss": max_monthly_loss,
            "Portfolio Turnover": portfolio_turnover
        }

    return results

def plot_q_value_heatmap(rl_agent, max_size=50):
    """
    Plot heatmap of Q-values, with handling for large Q-tables
    """
    q_table = rl_agent.q_table

    # If Q-table is too large, downsample it
    if q_table.shape[0] > max_size or q_table.shape[1] > max_size:
        row_step = max(1, q_table.shape[0] // max_size)
        col_step = max(1, q_table.shape[1] // max_size)
        q_table = q_table[::row_step, ::col_step]

    plt.figure(figsize=(12, 8))
    
    # Use non-annotated heatmap for very large tables
    if q_table.size > 1000:
        sns.heatmap(q_table, cmap='YlGnBu')
    else:
        sns.heatmap(q_table, cmap='YlGnBu', annot=True, fmt='.2f')
    
    plt.title('Q-value Heatmap (Downsampled)' if q_table.shape != rl_agent.q_table.shape else 'Q-value Heatmap')
    plt.xlabel('Actions (Portfolio Deciles)')
    plt.ylabel('States')
    plt.savefig('q_value_heatmap.png')
    plt.close()

    # Print Q-table statistics
    print(f"Q-table shape: {rl_agent.q_table.shape}")
    print(f"Q-table min value: {rl_agent.q_table.min():.4f}")
    print(f"Q-table max value: {rl_agent.q_table.max():.4f}")
    print(f"Q-table mean value: {rl_agent.q_table.mean():.4f}")

def plot_action_distribution(portfolios):
    """
    Plot distribution of actions (selected portfolios) over time
    """
    actions = [p['selected_portfolio'] for p in portfolios]
    
    plt.figure(figsize=(12, 6))
    plt.hist(actions, bins=range(11), align='left', rwidth=0.8)
    plt.title('Distribution of Selected Portfolios (Actions)')
    plt.xlabel('Portfolio Decile')
    plt.ylabel('Frequency')
    plt.xticks(range(10))
    plt.savefig('action_distribution.png')
    plt.close()

def plot_performance_metrics(portfolios):
    """
    Plot performance metrics (Sharpe ratio, returns, drawdowns) over time
    """
    rl_returns = pd.DataFrame([p['portfolio_returns'] for p in portfolios])
    
    # Calculate rolling metrics
    window = 12  # 1-year rolling window
    rolling_returns = rl_returns.rolling(window=window).mean()
    rolling_volatility = rl_returns.rolling(window=window).std()
    rolling_sharpe = (rolling_returns / rolling_volatility) * np.sqrt(12)
    rolling_drawdown = rl_returns.rolling(window=window).apply(lambda x: (1+x).cumprod().max() - (1+x).cumprod().iloc[-1])
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot rolling Sharpe ratio
    rolling_sharpe.plot(ax=axes[0])
    axes[0].set_title('Rolling Sharpe Ratio (1-year window)')
    axes[0].set_xlabel('')
    axes[0].legend(loc='best')
    
    # Plot rolling returns
    rolling_returns.plot(ax=axes[1])
    axes[1].set_title('Rolling Returns (1-year window)')
    axes[1].set_xlabel('')
    axes[1].legend(loc='best')
    
    # Plot rolling drawdowns
    rolling_drawdown.plot(ax=axes[2])
    axes[2].set_title('Rolling Drawdowns (1-year window)')
    axes[2].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.close()

def plot_hrp_dendrogram(returns, n_stocks=50):
    # Select n_stocks randomly
    selected_stocks = returns.sample(n=n_stocks, axis=1, random_state=42)
    
    # Create the dendrogram
    ax = rp.plot_dendrogram(returns=selected_stocks, 
                            codependence='pearson', 
                            linkage='single', 
                            k=None, 
                            max_k=10, 
                            leaf_order=True, 
                            ax=None)
    
    plt.title('HRP Dendrogram')
    plt.savefig('hrp_dendrogram.png')
    plt.close()

def plot_hrp_pie_chart(weights, title='HRP Portfolio Weights'):
    """
    Plot a pie chart of the HRP portfolio weights.
    """
    plt.figure(figsize=(10, 8))
    ax = rp.plot_pie(w=weights, 
                     title=title, 
                     others=0.05, 
                     nrow=25, 
                     cmap="tab20", 
                     height=8, 
                     width=10, 
                     ax=None)
    plt.savefig('hrp_pie_chart.png')
    plt.close()

def plot_hrp_risk_contribution(weights, returns, cov, rm='MV', rf=0):
    """
    Plot the risk contribution of each asset in the HRP portfolio.
    """
    plt.figure(figsize=(10, 6))
    ax = rp.plot_risk_con(w=weights, 
                          cov=cov, 
                          returns=returns, 
                          rm=rm, 
                          rf=rf, 
                          alpha=0.05, 
                          color="tab:blue", 
                          height=6, 
                          width=10, 
                          t_factor=252, 
                          ax=None)
    plt.title('HRP Risk Contribution')
    plt.savefig('hrp_risk_contribution.png')
    plt.close()

def plot_cumulative_returns1(portfolios):
    """
    Plot cumulative returns of the DQN strategy for each model type.
    """
    returns = pd.DataFrame([p['portfolio_returns'] for p in portfolios])
    cumulative_returns = (1 + returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column], label=column)
    
    plt.title('Cumulative Returns: DQN Strategy')
    plt.xlabel('Time Period')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('dqn_cumulative_returns.png')
    plt.close()

def plot_sharpe_ratios(portfolios):
    """
    Plot Sharpe ratios over time.
    """
    sharpe_ratios = [p['sharpe_ratio'] for p in portfolios]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sharpe_ratios)), sharpe_ratios)
    plt.title('Sharpe Ratio Over Time: DQN Strategy')
    plt.xlabel('Time Period')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.savefig('dqn_sharpe_ratios.png')
    plt.close()

def plot_action_distribution(portfolios):
    """
    Plot distribution of actions (selected portfolios) over time.
    """
    actions = [p['selected_portfolio'] for p in portfolios]
    
    plt.figure(figsize=(12, 6))
    sns.histplot(actions, kde=True, bins=range(min(actions), max(actions) + 2, 1))
    plt.title('Distribution of Selected Portfolios (Actions): DQN Strategy')
    plt.xlabel('Portfolio Decile')
    plt.ylabel('Frequency')
    plt.savefig('dqn_action_distribution.png')
    plt.close()

def plot_returns_heatmap(portfolios):
    """
    Plot a heatmap of returns for each model type over time.
    """
    returns = pd.DataFrame([p['portfolio_returns'] for p in portfolios])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(returns, cmap='YlGnBu', annot=True, fmt='.2f')
    plt.title('Returns Heatmap: DQN Strategy')
    plt.xlabel('Model Type')
    plt.ylabel('Time Period')
    plt.savefig('dqn_returns_heatmap.png')
    plt.close()

def plot_performance_comparison(portfolios):
    """
    Plot a comparison of performance metrics for each model type.
    """
    returns = pd.DataFrame([p['portfolio_returns'] for p in portfolios])
    
    metrics = {
        'Mean Return': returns.mean(),
        'Std Dev': returns.std(),
        'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252),  # Assuming daily returns
        'Max Drawdown': (returns.cummax() - returns).max()
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', ax=plt.gca())
    plt.title('Performance Metrics Comparison: DQN Strategy')
    plt.xlabel('Model Type')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('dqn_performance_comparison.png')
    plt.close()

def visualize_dqn_performance(portfolios):
    """
    Generate all visualizations for DQN performance.
    """
    plot_cumulative_returns(portfolios)
    plot_sharpe_ratios(portfolios)
    plot_action_distribution(portfolios)
    plot_returns_heatmap(portfolios)
    plot_performance_comparison(portfolios)
     
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
    
    #selected_features = pipeline.select_features(X, y, n_features=20)
    selected_features = ['cop_atl1', 'ope_bel1', 'lti_gr1a', 'sti_gr1a', 'rd5_at', 'ni_be', 'ebit_sale', 'cop_at', 'op_at', 'tangibility', 'fnl_gr1a', 'ocf_at', 'rd_sale', 'f_score', 'nfna_gr1a', 'qmj', 'niq_be', 'gp_at', 'debt_gr3', 'ni_inc8q']
   # print(f"Selected features: {selected_features}")
   # pipeline.plot_feature_importance(X, y)
    #NEW
   # importances = pipeline.compute_feature_importance(X, y)
   # print(importances)
   # pipeline.plot_feature_importances(importances)

    results = run_rolling_cross_validation(data, selected_features, ret_var, train_duration=120, test_duration=12, lookahead=1, n_splits=12)
    
    
    # Hirearchical Risk Parity Approach
    #hrp_portfolios = create_portfolio_with_hrp(results, n_stocks=50)

    # Calculate and print performance metrics
#    hrp_returns = [p['portfolio_return'] for p in hrp_portfolios]
#    df_returns = pd.DataFrame({'HRP': hrp_returns})
#    print(df_returns['HRP'])
#    sharpe_ratio = df_returns['HRP'].mean() / df_returns['HRP'].std() * np.sqrt(12)
#    cumulative_return = (1 + df_returns['HRP']).prod() - 1
#    max_drawdown = (df_returns['HRP'].cummax() - df_returns['HRP']).max()

#    print(f"Performance metrics for HRP:")
#    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
#    print(f"Cumulative Return: {cumulative_return:.4f}")
#    print(f"Max Drawdown: {max_drawdown:.4f}")
#    print("\n")

    # Plot HRP dendrogram
   # all_returns = pd.concat([r['predictions'].pivot(index='date', columns='stock_ticker', values='actual_return') for r in results])
   # plot_hrp_dendrogram(all_returns, n_stocks=50)
    
  #  last_portfolio = hrp_portfolios[-1]
  #  weights = last_portfolio['portfolio_allocations']
  #  selected_stocks = last_portfolio['selected_stocks']

#    plot_hrp_pie_chart(weights, title='HRP Portfolio Weights (Last Period)')

 #   mu = selected_stocks.mean()
  #  cov = selected_stocks.cov()
   # plot_hrp_risk_contribution(weights, returns=selected_stocks, cov=cov, rm='MV', rf=0)

    # Save results
    #save_results(results, hrp_portfolios, "HRP_results")

    # Generate visualizations
    #plot_cumulative_returns(hrp_portfolios)
    
    
    # RL Approach
    n_states = 2 ** 20  # 4 market indicators
    rl_agent = RLAgent(n_actions=10, n_states=n_states)
   # rl_agent = RLAgent(n_portfolios=10, n_stocks=len(stock_vars)) #v1
    portfolios_Rl, rl_agent = create_portfolio_with_rl(results, n_portfolios=10, rl_agent=rl_agent)

    #portfolios_Rl, rl_agent = create_portfolio_with_rl(results, n_portfolios=10, n_stocks=len(stock_vars), rl_agent=rl_agent) #v1
    print("RL PORT: ", portfolios_Rl)
    print("AGENT:", rl_agent)

    plot_cumulative_returns(portfolios_Rl)
    plot_q_value_heatmap(rl_agent)
    plot_action_distribution(portfolios_Rl)
    plot_performance_metrics(portfolios_Rl)
    #calculate_performance_metrics(portfolios_Rl)
    save_results(results,portfolios_Rl, "RL_results")
    
    portfolios_DQN, dqn_agent = create_portfolio_with_dqn(results)

    calculate_performance_metrics3(portfolios_DQN, is_dqn=True)
    visualize_dqn_performance(portfolios_DQN)
    print("---")
    # Vanilla Approach
    portfolios = create_portfolio(results)

    portfolio_returns = calculate_portfolio_returns(portfolios)

    calculate_performance_metrics1(portfolio_returns)

    output_dir = "results"
    save_results(results, portfolios, output_dir)

    print(datetime.datetime.now())
