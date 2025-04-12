import unittest
import pandas as pd
import numpy as np
from easyda import MVPFeatureSelector

class TestMVPFeatureSelector(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        np.random.seed(42)
        X = np.random.rand(100, 5)
        # Make feature1 highly correlated with target
        y = X[:, 0] * 3 + X[:, 1] - X[:, 2] * 0.5 + X[:, 3] * 0.1 + np.random.randn(100) * 0.1
        
        self.data = pd.DataFrame(
            np.column_stack([X, y]), 
            columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'target']
        )
        
        self.selector = MVPFeatureSelector(self.data)
    
    def test_init(self):
        # Test initialization with data
        self.assertTrue(isinstance(self.selector.data, pd.DataFrame))
        self.assertEqual(len(self.selector.data), 100)
        
        # Test initialization without data
        empty_selector = MVPFeatureSelector()
        self.assertIsNone(empty_selector.data)
    
    def test_set_data(self):
        # Test setting data
        new_data = pd.DataFrame({'X': [1, 2, 3], 'y': [10, 20, 30]})
        self.selector.set_data(new_data)
        self.assertEqual(list(self.selector.data.columns), ['X', 'y'])
        self.assertEqual(len(self.selector.data), 3)
    
    def test_select_best_features_f_test(self):
        # Test feature selection with f_test
        result = self.selector.select_best_features('target', n_features=3, method='f_test')
        self.assertTrue('selected_features' in result)
        self.assertTrue('all_scores' in result)
        self.assertEqual(len(result['selected_features']), 3)
        self.assertEqual(len(result['all_scores']), 5)  # 5 features total
        
        # Feature1 should be in the selected features (it's highly correlated with target)
        self.assertTrue('feature1' in result['selected_features'])
    
    def test_select_best_features_mutual_info(self):
        # Test feature selection with mutual_info
        result = self.selector.select_best_features('target', n_features=3, method='mutual_info')
        self.assertTrue('selected_features' in result)
        self.assertTrue('all_scores' in result)
        self.assertEqual(len(result['selected_features']), 3)
        self.assertEqual(len(result['all_scores']), 5)  # 5 features total
        
        # Feature1 should be in the selected features (it's highly correlated with target)
        self.assertTrue('feature1' in result['selected_features'])
    
    def test_invalid_method(self):
        # Test with invalid method
        with self.assertRaises(ValueError):
            self.selector.select_best_features('target', method='invalid_method')

if __name__ == '__main__':
    unittest.main()
