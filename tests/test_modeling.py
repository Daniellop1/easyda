import unittest
import pandas as pd
import numpy as np
from easyda import ModelBuilder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y_reg = X[:, 0] * 2 + X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
        y_cls = (y_reg > 0.5).astype(int)
        
        self.data_reg = pd.DataFrame(
            np.column_stack([X, y_reg]), 
            columns=['feature1', 'feature2', 'feature3', 'target']
        )
        
        self.data_cls = pd.DataFrame(
            np.column_stack([X, y_cls]), 
            columns=['feature1', 'feature2', 'feature3', 'target']
        )
        
        self.model_builder_reg = ModelBuilder(self.data_reg)
        self.model_builder_cls = ModelBuilder(self.data_cls)
    
    def test_init(self):
        # Test initialization with data
        self.assertTrue(isinstance(self.model_builder_reg.data, pd.DataFrame))
        self.assertEqual(len(self.model_builder_reg.data), 100)
        
        # Test initialization without data
        empty_model_builder = ModelBuilder()
        self.assertTrue(isinstance(empty_model_builder.data, pd.DataFrame))
        self.assertEqual(len(empty_model_builder.data), 0)
    
    def test_set_data(self):
        # Test setting data
        new_data = pd.DataFrame({'X': [1, 2, 3], 'y': [10, 20, 30]})
        self.model_builder_reg.set_data(new_data)
        self.assertEqual(list(self.model_builder_reg.data.columns), ['X', 'y'])
        self.assertEqual(len(self.model_builder_reg.data), 3)
    
    def test_prepare_data(self):
        # Test data preparation for regression
        self.model_builder_reg.prepare_data(target='target')
        self.assertEqual(self.model_builder_reg.X_train.shape[1], 3)
        self.assertTrue(hasattr(self.model_builder_reg, 'y_train'))
        self.assertTrue(hasattr(self.model_builder_reg, 'X_test'))
        self.assertTrue(hasattr(self.model_builder_reg, 'y_test'))
        
        # Test data preparation with specific features
        self.model_builder_reg.prepare_data(target='target', features=['feature1', 'feature2'])
        self.assertEqual(self.model_builder_reg.X_train.shape[1], 2)
    
    def test_train_regression(self):
        # Test training a regression model
        self.model_builder_reg.prepare_data(target='target')
        self.model_builder_reg.train(model_type='linear_regression')
        self.assertTrue(isinstance(self.model_builder_reg.model, LinearRegression))
        
        self.model_builder_reg.train(model_type='random_forest_regressor')
        self.assertTrue(isinstance(self.model_builder_reg.model, RandomForestRegressor))
    
    def test_train_classification(self):
        # Test training a classification model
        self.model_builder_cls.prepare_data(target='target')
        self.model_builder_cls.train(model_type='logistic_regression')
        self.assertTrue(isinstance(self.model_builder_cls.model, LogisticRegression))
        
        self.model_builder_cls.train(model_type='random_forest_classifier')
        self.assertTrue(isinstance(self.model_builder_cls.model, RandomForestClassifier))
    
    def test_predict_evaluate_regression(self):
        # Test prediction and evaluation for regression
        self.model_builder_reg.prepare_data(target='target')
        self.model_builder_reg.train(model_type='linear_regression')
        predictions = self.model_builder_reg.predict()
        self.assertEqual(len(predictions), len(self.model_builder_reg.y_test))
        
        evaluation = self.model_builder_reg.evaluate()
        self.assertTrue('mean_squared_error' in evaluation)
        self.assertTrue('r2_score' in evaluation)
    
    def test_predict_evaluate_classification(self):
        # Test prediction and evaluation for classification
        self.model_builder_cls.prepare_data(target='target')
        self.model_builder_cls.train(model_type='logistic_regression')
        predictions = self.model_builder_cls.predict()
        self.assertEqual(len(predictions), len(self.model_builder_cls.y_test))
        
        evaluation = self.model_builder_cls.evaluate()
        self.assertTrue('accuracy' in evaluation)

if __name__ == '__main__':
    unittest.main()
