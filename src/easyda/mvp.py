import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .data_processing import DataProcessor
from .visualization import Visualizer
from .modeling import ModelBuilder

class MVPAnalyzer:
    """Class for quick MVP data analysis with minimal code."""
    
    def __init__(self, data=None, filepath=None):
        """Initialize with data or load from file."""
        self.data_processor = DataProcessor()
        
        if filepath:
            self.data_processor.load_csv(filepath)
        elif data is not None:
            self.data_processor.data = data
            
        self.visualizer = Visualizer(self.data_processor.data)
        self.model_builder = ModelBuilder(self.data_processor.data)
        self.insights = {}
        
    def quick_analysis(self, target_column=None):
        """Perform a quick comprehensive analysis of the dataset."""
        # Data summary
        self.insights['summary'] = self.data_processor.summary()
        
        # Data cleaning
        self.data_processor.clean()
        
        # Visualizations
        numeric_columns = self.data_processor.data.select_dtypes(include=[np.number]).columns
        
        # Create visualizations for numeric columns
        for column in numeric_columns[:5]:  # Limit to first 5 columns to avoid too many plots
            self.visualizer.histogram(column)
            
        # Correlation analysis
        self.visualizer.correlation_heatmap()
        
        # If target column is provided, perform predictive modeling
        if target_column and target_column in self.data_processor.data.columns:
            features = [col for col in numeric_columns if col != target_column]
            
            # Check if target is categorical or continuous
            if self.data_processor.data[target_column].dtype == 'object' or len(self.data_processor.data[target_column].unique()) < 10:
                # Classification
                self.model_builder.prepare_data(target=target_column, features=features)
                self.model_builder.train(model_type='random_forest_classifier')
                self.insights['model_evaluation'] = self.model_builder.evaluate()
            else:
                # Regression
                self.model_builder.prepare_data(target=target_column, features=features)
                self.model_builder.train(model_type='random_forest_regressor')
                self.insights['model_evaluation'] = self.model_builder.evaluate()
                
        return self.insights
    
    def feature_importance(self, target_column):
        """Analyze and visualize feature importance for the target."""
        if not hasattr(self.model_builder, 'model') or self.model_builder.model is None:
            features = self.data_processor.data.select_dtypes(include=[np.number]).columns
            features = [col for col in features if col != target_column]
            self.model_builder.prepare_data(target=target_column, features=features)
            self.model_builder.train(model_type='random_forest_regressor')
            
        if hasattr(self.model_builder.model, 'feature_importances_'):
            importances = self.model_builder.model.feature_importances_
            feature_names = self.model_builder.X_train.columns if hasattr(self.model_builder.X_train, 'columns') else [f"Feature {i}" for i in range(len(importances))]
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()
            
            return {feature_names[i]: importances[i] for i in indices}
        else:
            return {"error": "Model does not support feature importance"}
