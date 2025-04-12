import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Visualizer:
    """Class for simplified data visualization."""
    
    def __init__(self, data=None):
        """Initialize with optional data."""
        self.data = data if data is not None else pd.DataFrame()
    
    def set_data(self, data):
        """Set the data to visualize."""
        self.data = data
        return self
    
    def histogram(self, column, bins=10, title=None, figsize=(10, 6)):
        """Create a histogram for a numeric column."""
        plt.figure(figsize=figsize)
        plt.hist(self.data[column], bins=bins)
        plt.title(title or f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        return self
    
    def scatter(self, x_col, y_col, title=None, figsize=(10, 6)):
        """Create a scatter plot between two columns."""
        plt.figure(figsize=figsize)
        plt.scatter(self.data[x_col], self.data[y_col])
        plt.title(title or f'Scatter plot of {y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.show()
        return self
    
    def correlation_heatmap(self, figsize=(12, 10)):
        """Create a correlation heatmap for numeric columns."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        
        plt.figure(figsize=figsize)
        plt.imshow(corr, cmap='coolwarm')
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        return self
