import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import warnings


warnings.filterwarnings('ignore')

def load_data(filepath):
    
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def clean_and_preprocess_data(df):
    
    if df is None:
        return None, None, None, None


    if 'Total Engagement' not in df.columns:
        
        cols = ['likes', 'comments', 'shares', 'saves']
        existing_cols = [c for c in cols if c in df.columns]
        if existing_cols:
             df['Total Engagement'] = df[existing_cols].sum(axis=1)
        else:
             raise ValueError("Target variable 'Total Engagement' not found and cannot be calculated.")

    target = 'Total Engagement'

    
    drop_cols = [
        'post_id', 'upload_date', 
        'likes', 'comments', 'shares', 'saves', 
        'followers_gained', 'reach', 'impressions', 
        'engagement_rate', 'traffic_source'
    ]
    

    drop_cols = [c for c in drop_cols if c in df.columns]
    df_clean = df.drop(columns=drop_cols)
    print(f"Columns dropped: {drop_cols}")

    
    df_clean = df_clean.dropna(subset=[target])
    
    
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        

    num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    
    feature_num_cols = [c for c in num_cols if c != target]
    for col in feature_num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    print("Missing values handled.")

    
    features_cat = [c for c in cat_cols if c in df_clean.columns]
    
    df_processed = pd.get_dummies(df_clean, columns=features_cat, drop_first=True)
    print(f"Categorical variables encoded. New shape: {df_processed.shape}")

    
    X = df_processed.drop(columns=[target])
    y = df_processed[target]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    print("Data split and scaled.")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
   
    df = load_data(r'C:\Users\sath2\OneDrive\Desktop\project\Instagram_Analytics_PowerBI_Ready.csv')
    if df is not None:
        X_train, X_test, y_train, y_test, feature_names = clean_and_preprocess_data(df)
        print("Data Processing Test Complete.")
