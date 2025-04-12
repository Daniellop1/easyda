import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from easyda import Visualizer

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['x', 'y', 'z', 'x', 'y']
        })
        self.visualizer = Visualizer(self.data)
    
    def test_init(self):
        # Test initialization with data
        self.assertTrue(isinstance(self.visualizer.data, pd.DataFrame))
        self.assertEqual(len(self.visualizer.data), 5)
        
        # Test initialization without data
        empty_visualizer = Visualizer()
        self.assertTrue(isinstance(empty_visualizer.data, pd.DataFrame))
        self.assertEqual(len(empty_visualizer.data), 0)
    
    def test_set_data(self):
        # Test setting data
        new_data = pd.DataFrame({'X': [1, 2, 3]})
        self.visualizer.set_data(new_data)
        self.assertEqual(list(self.visualizer.data.columns), ['X'])
        self.assertEqual(len(self.visualizer.data), 3)
    
    def test_histogram(self):
        # Test histogram creation (just check if it runs without errors)
        try:
            # Turn off interactive mode to avoid displaying plots during tests
            plt.ioff()
            result = self.visualizer.histogram('A')
            self.assertEqual(result, self.visualizer)
        finally:
            plt.close('all')
    
    def test_scatter(self):
        # Test scatter plot creation
        try:
            plt.ioff()
            result = self.visualizer.scatter('A', 'B')
            self.assertEqual(result, self.visualizer)
        finally:
            plt.close('all')
    
    def test_correlation_heatmap(self):
        # Test correlation heatmap creation
        try:
            plt.ioff()
            result = self.visualizer.correlation_heatmap()
            self.assertEqual(result, self.visualizer)
        finally:
            plt.close('all')

if __name__ == '__main__':
    unittest.main()
