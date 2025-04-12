import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_regression, mutual_info_classif

class MVPFeatureSelector:
    """Class for selecting the most important features for MVP models."""
    
    def __init__(self, data=None):
        """Initialize with optional data."""
        self.data = data
        
    def set_data(self, data):
        """Set the data for feature selection."""
        self.data = data
        return self
    
    def select_best_features(self, target, n_features=5, method='f_test'):
        """Select the best features for predicting the target variable."""
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")
            
        X = self.data.drop(target, axis=1)
        y = self.data[target]
        
        # Only use numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Choose the appropriate scoring function
        if method == 'f_test':
            if len(np.unique(y)) < 5:  # Classification
                score_func = f_classif
            else:  # Regression
                score_func = f_regression
        elif method == 'mutual_info':
            if len(np.unique(y)) < 5:  # Classification
                score_func = mutual_info_classif
            else:  # Regression
                score_func = mutual_info_regression
        else:
            raise ValueError("Method must be 'f_test' or 'mutual_info'")
        
        # Select best features
        selector = SelectKBest(score_func=score_func, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected feature names
        mask = selector.get_support()
        selected_features = X.columns[mask].tolist()
        
        # Get scores
        scores = selector.scores_
        feature_scores = {X.columns[i]: scores[i] for i in range(len(X.columns))}
        
        return {
            'selected_features': selected_features,
            'all_scores': feature_scores
        }
