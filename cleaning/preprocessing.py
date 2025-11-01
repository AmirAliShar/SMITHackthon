import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def validate_dataframe_numeric_only(df):
    non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
    return (len(non_numeric) == 0), non_numeric

def preprocess_data(df, task_type="supervised", target_column=None, scale_data=True):
    
    if task_type not in ["supervised", "unsupervised"]:
        raise ValueError("task_type must be 'supervised' or 'unsupervised'")

    df = df.copy()

    # Supervised needs target
    if task_type == "supervised":
        if target_column is None:
            raise ValueError("target_column required for supervised task")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle target column with mixed types like ['nan', '>300', 'Norm', '>200']
        if y.dtype == 'object':
            # Convert string 'nan' to actual NaN
            y = y.replace('nan', np.nan)
            
            # Try to convert to numeric, handling special cases
            try:
                # For values like '>300', '>200', we need custom handling
                # Convert to numeric, coercing errors to NaN
                y_numeric = pd.to_numeric(y, errors='coerce')
                
                # If we have too many NaN after conversion, use LabelEncoder
                if y_numeric.isna().sum() / len(y) > 0.5:
                    # Too many non-numeric values, use LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                else:
                    # Mostly numeric, use the numeric version
                    y = y_numeric
            except:
                # Fallback to LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
    
    else:
        X = df.copy()
        y = None

    # Detect numeric & categorical columns
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Fill missing values in features
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy='mean')
        X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
    
    # Fix for FutureWarning: use explicit dtype conversion
    for col in categorical_cols:
        mode_val = X[col].mode()
        fill_value = mode_val[0] if not mode_val.empty else "missing"
        # Convert to string explicitly to avoid downcasting warning
        X[col] = X[col].fillna(fill_value).astype(str)

    # Use LabelEncoder for ALL categorical columns
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Scale all numeric features
    if scale_data and len(X.columns) > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    if task_type == "supervised":
        # Handle missing values in target - convert to pandas Series if needed
        if hasattr(y, 'isna'): 
            if y.isna().any():
                if y.dtype in ['float64', 'int64'] or np.issubdtype(y.dtype, np.number):
                    y = y.fillna(y.mean())
                else:
                    y = y.fillna(y.mode()[0] if not y.mode().empty else 0)
        else:  # It's a numpy array
            # Convert to pandas Series to handle NaN properly
            y_series = pd.Series(y)
            if y_series.isna().any():
                # Check if numeric
                try:
                    y_series = pd.to_numeric(y_series, errors='coerce')
                    if y_series.isna().any():
                        y_series = y_series.fillna(y_series.mean())
                except:
                    y_series = y_series.fillna(y_series.mode()[0] if not y_series.mode().empty else 0)
            y = y_series.values  # Convert back to numpy array
        
        # Determine if classification or regression
        unique_values = len(np.unique(y))
        total_samples = len(y)
        
        
        is_classification = (unique_values / total_samples < 0.1) and (unique_values > 1) and (unique_values <= 20)
        
        if is_classification and unique_values >= 2:
            try:

                y_strat = y.astype(int) if not np.issubdtype(y.dtype, np.integer) else y
                strat = y_strat
            except:
                strat = None
        else:
            
            strat = None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )
        return X_train, X_test, y_train, y_test
    else:
        return X