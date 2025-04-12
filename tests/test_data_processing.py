import unittest
import pandas as pd
import numpy as np
from easyda import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        self.data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, 20, 30, 40, 10],
            'C': ['x', 'y', 'z', 'x', 'y']
        })
        self.processor = DataProcessor(self.data)
    
    def test_init(self):
        # Test initialization with data
        self.assertTrue(isinstance(self.processor.data, pd.DataFrame))
        self.assertEqual(len(self.processor.data), 5)
        
        # Test initialization without data
        empty_processor = DataProcessor()
        self.assertTrue(isinstance(empty_processor.data, pd.DataFrame))
        self.assertEqual(len(empty_processor.data), 0)
    
    def test_clean(self):
        # Test cleaning with default parameters
        cleaned = self.processor.clean()
        self.assertEqual(len(cleaned.data), 4)  # One row with NaN should be removed
        
        # Test cleaning with drop_na=False
        processor = DataProcessor(self.data)
        cleaned = processor.clean(drop_na=False)
        self.assertEqual(len(cleaned.data), 5)  # No rows should be removed
        
        # Test cleaning with drop_duplicates=False
        processor = DataProcessor(pd.DataFrame({
            'A': [1, 1, 2],
            'B': [10, 10, 20]
        }))
        cleaned = processor.clean(drop_duplicates=False)
        self.assertEqual(len(cleaned.data), 3)  # No rows should be removed
        
        # Test cleaning with drop_duplicates=True
        cleaned = processor.clean(drop_duplicates=True)
        self.assertEqual(len(cleaned.data), 2)  # Duplicate row should be removed
    
    def test_select(self):
        # Test selecting columns
        selected = self.processor.select(['A', 'B'])
        self.assertEqual(list(selected.data.columns), ['A', 'B'])
        self.assertEqual(len(selected.data), 5)
    
    def test_summary(self):
        # Test summary generation
        summary = self.processor.summary()
        self.assertEqual(summary['shape'], (5, 3))
        self.assertEqual(summary['columns'], ['A', 'B', 'C'])
        self.assertEqual(summary['missing_values']['A'], 1)
        self.assertEqual(summary['missing_values']['B'], 0)

if __name__ == '__main__':
    unittest.main()
