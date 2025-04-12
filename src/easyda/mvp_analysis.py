import matplotlib.pyplot as plt
import numpy as np

class ValueComplexityAnalyzer:
    """Analyze features based on their value and complexity for MVP development."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.features = []
        self.values = []
        self.complexities = []
        
    def add_feature(self, feature_name, value, complexity):
        """Add a feature with its value and complexity scores (1-10)."""
        self.features.append(feature_name)
        self.values.append(value)
        self.complexities.append(complexity)
        return self
    
    def plot_matrix(self):
        """Plot the value vs. complexity matrix."""
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(self.complexities, self.values, s=100)
        
        # Add feature labels
        for i, feature in enumerate(self.features):
            plt.annotate(feature, (self.complexities[i], self.values[i]), 
                         xytext=(5, 5), textcoords='offset points')
        
        # Add quadrant lines
        plt.axvline(x=5, color='gray', linestyle='--', alpha=0.7)
        plt.axhline(y=5, color='gray', linestyle='--', alpha=0.7)
        
        # Add quadrant labels
        plt.text(2.5, 7.5, "High Value\nLow Complexity\n(MVP Priority)", 
                 ha='center', va='center', bbox=dict(facecolor='green', alpha=0.1))
        plt.text(7.5, 7.5, "High Value\nHigh Complexity\n(Consider for MVP)", 
                 ha='center', va='center', bbox=dict(facecolor='yellow', alpha=0.1))
        plt.text(2.5, 2.5, "Low Value\nLow Complexity\n(Nice to have)", 
                 ha='center', va='center', bbox=dict(facecolor='blue', alpha=0.1))
        plt.text(7.5, 2.5, "Low Value\nHigh Complexity\n(Avoid for MVP)", 
                 ha='center', va='center', bbox=dict(facecolor='red', alpha=0.1))
        
        # Set labels and title
        plt.xlabel('Complexity (1-10)')
        plt.ylabel('Value (1-10)')
        plt.title('Value vs. Complexity Analysis for MVP Features')
        
        # Set axis limits
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def get_mvp_recommendations(self):
        """Get recommendations for MVP features based on value-complexity analysis."""
        recommendations = {
            'high_priority': [],
            'consider': [],
            'nice_to_have': [],
            'avoid': []
        }
        
        for i, feature in enumerate(self.features):
            value = self.values[i]
            complexity = self.complexities[i]
            
            if value >= 5 and complexity < 5:
                recommendations['high_priority'].append(feature)
            elif value >= 5 and complexity >= 5:
                recommendations['consider'].append(feature)
            elif value < 5 and complexity < 5:
                recommendations['nice_to_have'].append(feature)
            else:
                recommendations['avoid'].append(feature)
                
        return recommendations
