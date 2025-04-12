import unittest
import matplotlib.pyplot as plt
from easyda import ValueComplexityAnalyzer

class TestValueComplexityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ValueComplexityAnalyzer()
        
        # Add some features
        self.analyzer.add_feature("Feature 1", 8, 3)  # High value, low complexity
        self.analyzer.add_feature("Feature 2", 9, 7)  # High value, high complexity
        self.analyzer.add_feature("Feature 3", 4, 2)  # Low value, low complexity
        self.analyzer.add_feature("Feature 4", 3, 8)  # Low value, high complexity
    
    def test_add_feature(self):
        # Test adding a feature
        self.analyzer.add_feature("Feature 5", 5, 5)
        self.assertEqual(len(self.analyzer.features), 5)
        self.assertEqual(self.analyzer.features[-1], "Feature 5")
        self.assertEqual(self.analyzer.values[-1], 5)
        self.assertEqual(self.analyzer.complexities[-1], 5)
    
    def test_plot_matrix(self):
        # Test plotting (just check if it runs without errors)
        try:
            plt.ioff()
            self.analyzer.plot_matrix()
        finally:
            plt.close('all')
    
    def test_get_mvp_recommendations(self):
        # Test getting recommendations
        recommendations = self.analyzer.get_mvp_recommendations()
        
        self.assertTrue('high_priority' in recommendations)
        self.assertTrue('consider' in recommendations)
        self.assertTrue('nice_to_have' in recommendations)
        self.assertTrue('avoid' in recommendations)
        
        # Check if features are in the right categories
        self.assertTrue("Feature 1" in recommendations['high_priority'])
        self.assertTrue("Feature 2" in recommendations['consider'])
        self.assertTrue("Feature 3" in recommendations['nice_to_have'])
        self.assertTrue("Feature 4" in recommendations['avoid'])

if __
