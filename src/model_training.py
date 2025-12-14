from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import warnings


warnings.filterwarnings('ignore')

def train_linear_regression(X_train, y_train):
   
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    
    print("Training Gradient Boosting Regressor (XGBoost)...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_all_models(X_train, y_train):
   
    models = {}
    models['Linear Regression'] = train_linear_regression(X_train, y_train)
    models['Random Forest'] = train_random_forest(X_train, y_train)
    models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train)
    
    print("All models trained successfully.")
    return models
