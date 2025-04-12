import unittest
import pandas as pd
import numpy as np
from easyda import MVPAnalyzer

class TestMVPAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y_reg = X[:, 0] * 2 + X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
        
        self.data = pd.DataFrame(
            np.column_stack([X, y_reg]), 
            columns=['feature1', 'feature2', 'feature3', 'target']
        )
        
        self.analyzer = MVPAnalyzer(self.data)
    
    def test_init(self):
        # Test initialization with data
        self.assertTrue(isinstance(self.analyzer.data_processor.data, pd.DataFrame))
        self.assertEqual(len(self.analyzer.data_processor.data), 100)
        
        # Test initialization with filepath
        # This would require a mock for file operations, skipping for simplicity
        
        # Test initialization without data
        empty_analyzer = MVPAnalyzer()
        self.assertTrue(isinstance(empty_analyzer.data_processor.data, pd.DataFrame))
        self.assertEqual(len(empty_analyzer.data_processor.data), 0)
    
    def test_quick_analysis(self):
        # Test quick analysis without target
        insights = self.analyzer.quick_analysis()
        self.assertTrue('summary' in insights)
        
        # Test quick analysis with target
        insights = self.analyzer.quick_analysis(target_column='target')
        self.assertTrue('summary' in insights)
        self.assertTrue('model_evaluation' in insights)
        
        # For regression, should have mean_squared_error and r2_score
        self.assertTrue('mean_squared_error' in insights['model_evaluation'])
        self.assertTrue('r2_score' in insights['model_evaluation'])
    
    def test_feature_importance(self):
        # Test feature importance
        importances = self.analyzer.feature_importance('target')
        self.assertTrue(isinstance(importances, dict))
        self.assertEqual(len(importances), 3)  # 3 features

if __name__ == '__main__':
    unittest.main()
