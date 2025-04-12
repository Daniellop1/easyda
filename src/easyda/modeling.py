import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

class ModelBuilder:
    """Class for simplified machine learning modeling."""
    
    def __init__(self, data=None):
        """Initialize with optional data."""
        self.data = data if data is not None else pd.DataFrame()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.predictions = None
        
    def set_data(self, data):
        """Set the data to use for modeling."""
        self.data = data
        return self
    
    def prepare_data(self, target, features=None, test_size=0.2, random_state=42, scale=False):
        """Prepare data for modeling by splitting into train and test sets."""
        if features is None:
            features = [col for col in self.data.columns if col != target]
            
        X = self.data[features]
        y = self.data[target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if scale:
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            
        return self
    
    def train(self, model_type='linear_regression', **kwargs):
        """Train a model on the prepared data."""
        model_types = {
            'linear_regression': LinearRegression,
            'logistic_regression': LogisticRegression,
            'random_forest_classifier': RandomForestClassifier,
            'random_forest_regressor': RandomForestRegressor
        }
        
        if model_type not in model_types:
            raise ValueError(f"Model type '{model_type}' not supported. Choose from: {list(model_types.keys())}")
            
        self.model = model_types[model_type](**kwargs)
        self.model.fit(self.X_train, self.y_train)
        return self
    
    def predict(self):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("No model has been trained yet. Call train() first.")
            
        self.predictions = self.model.predict(self.X_test)
        return self.predictions
    
    def evaluate(self):
        """Evaluate the model's performance."""
        if self.predictions is None:
            self.predict()
            
        # Determine if classification or regression
        if hasattr(self.model, 'predict_proba'):
            # Classification
            accuracy = accuracy_score(self.y_test, self.predictions)
            return {'accuracy': accuracy}
        else:
            # Regression
            mse = mean_squared_error(self.y_test, self.predictions)
            r2 = r2_score(self.y_test, self.predictions)
            return {'mean_squared_error': mse, 'r2_score': r2}
