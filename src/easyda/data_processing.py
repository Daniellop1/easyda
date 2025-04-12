import pandas as pd
import numpy as np

class DataProcessor:
    """Class for simplified data processing operations."""
    
    def __init__(self, data=None):
        """Initialize with optional data."""
        self.data = data if data is not None else pd.DataFrame()
    
    def load_csv(self, filepath, **kwargs):
        """Load data from CSV file."""
        self.data = pd.read_csv(filepath, **kwargs)
        return self
    
    def clean(self, columns=None, drop_na=True, drop_duplicates=True):
        """Clean the dataset by handling missing values and duplicates."""
        if columns is None:
            columns = self.data.columns
            
        if drop_na:
            self.data = self.data.dropna(subset=columns)
        
        if drop_duplicates:
            self.data = self.data.drop_duplicates(subset=columns)
            
        return self
    
    def select(self, columns):
        """Select specific columns."""
        self.data = self.data[columns]
        return self
    
    def summary(self):
        """Return a summary of the dataset."""
        summary = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "numeric_summary": self.data.describe().to_dict() if not self.data.empty else {}
        }
        return summary
