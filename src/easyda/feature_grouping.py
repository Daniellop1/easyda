import pandas as pd
import numpy as np

class FeatureGrouper:
    """Group features by their numeric value/importance."""
    
    def __init__(self, feature_scores=None):
        """Initialize with optional feature scores."""
        self.feature_scores = feature_scores or {}
        
    def set_feature_scores(self, feature_scores):
        """Set the feature scores dictionary."""
        self.feature_scores = feature_scores
        return self
    
    def group_by_percentile(self, n_groups=3):
        """Group features into n groups based on percentiles."""
        if not self.feature_scores:
            raise ValueError("Feature scores not set. Call set_feature_scores() first.")
            
        scores = np.array(list(self.feature_scores.values()))
        features = list(self.feature_scores.keys())
        
        # Calculate percentile thresholds
        thresholds = [np.percentile(scores, 100 * i / n_groups) for i in range(1, n_groups)]
        
        groups = {f"Group {i+1}": [] for i in range(n_groups)}
        
        # Assign features to groups
        for feature, score in self.feature_scores.items():
            group_idx = n_groups - 1
            for i, threshold in enumerate(thresholds):
                if score <= threshold:
                    group_idx = i
                    break
            groups[f"Group {group_idx+1}"].append(feature)
            
        return groups
    
    def group_by_value(self, thresholds):
        """Group features based on custom thresholds."""
        if not self.feature_scores:
            raise ValueError("Feature scores not set. Call set_feature_scores() first.")
            
        n_groups = len(thresholds) + 1
        groups = {f"Group {i+1}": [] for i in range(n_groups)}
        
        # Assign features to groups
        for feature, score in self.feature_scores.items():
            group_idx = n_groups - 1
            for i, threshold in enumerate(thresholds):
                if score <= threshold:
                    group_idx = i
                    break
            groups[f"Group {group_idx+1}"].append(feature)
            
        return groups
