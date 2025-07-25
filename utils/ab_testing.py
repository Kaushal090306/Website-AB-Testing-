import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns

class ABTestAnalyzer:
    """
    A comprehensive A/B testing analysis class
    """
    
    def __init__(self, control_conversions, control_visitors, 
                 treatment_conversions, treatment_visitors, alpha=0.05):
        self.control_conversions = control_conversions
        self.control_visitors = control_visitors
        self.treatment_conversions = treatment_conversions
        self.treatment_visitors = treatment_visitors
        self.alpha = alpha
        
        # Calculate basic metrics
        self.control_rate = control_conversions / control_visitors
        self.treatment_rate = treatment_conversions / treatment_visitors
        self.lift = self.treatment_rate - self.control_rate
        self.relative_lift = (self.lift / self.control_rate) * 100 if self.control_rate > 0 else 0
        
    def proportion_ci(self, successes, n, alpha=None):
        """Calculate confidence interval for proportion"""
        if alpha is None:
            alpha = self.alpha
            
        p_hat = successes / n
        z = stats.norm.ppf(1 - alpha/2)
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        return p_hat, p_hat - z*se, p_hat + z*se
    
    def z_test(self):
        """Perform two-proportion z-test"""
        count = np.array([self.treatment_conversions, self.control_conversions])
        nobs = np.array([self.treatment_visitors, self.control_visitors])
        z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')
        return z_stat, p_value
    
    def effect_size(self):
        """Calculate Cohen's d for proportions"""
        pooled_p = (self.control_conversions + self.treatment_conversions) / \
                   (self.control_visitors + self.treatment_visitors)
        pooled_se = np.sqrt(pooled_p * (1 - pooled_p) * 
                           (1/self.control_visitors + 1/self.treatment_visitors))
        return self.lift / pooled_se if pooled_se > 0 else 0
    
    def power_analysis(self, power=0.8):
        """Estimate required sample size for given power"""
        effect_size = self.effect_size()
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2 if effect_size > 0 else float('inf')
        return n
    
    def get_summary(self):
        """Get comprehensive test summary"""
        z_stat, p_value = self.z_test()
        effect_size = self.effect_size()
        
        control_ci = self.proportion_ci(self.control_conversions, self.control_visitors)
        treatment_ci = self.proportion_ci(self.treatment_conversions, self.treatment_visitors)
        
        return {
            'control_rate': self.control_rate,
            'treatment_rate': self.treatment_rate,
            'lift': self.lift,
            'relative_lift': self.relative_lift,
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size': effect_size,
            'control_ci': control_ci,
            'treatment_ci': treatment_ci,
            'required_sample_size': self.power_analysis()
        }

def simulate_ab_test(n_control, n_treatment, p_control, p_treatment, seed=42):
    """
    Simulate A/B test data
    """
    np.random.seed(seed)
    
    conversions_control = np.random.binomial(n_control, p_control)
    conversions_treatment = np.random.binomial(n_treatment, p_treatment)
    
    return conversions_control, conversions_treatment

def sequential_testing_simulation(n_batches, batch_size, p_control, p_treatment, seed=42):
    """
    Simulate sequential A/B testing
    """
    np.random.seed(seed)
    
    results = []
    total_control_visitors = 0
    total_treatment_visitors = 0
    total_control_conversions = 0
    total_treatment_conversions = 0
    
    for batch in range(1, n_batches + 1):
        # Simulate new batch
        new_control_conversions = np.random.binomial(batch_size, p_control)
        new_treatment_conversions = np.random.binomial(batch_size, p_treatment)
        
        # Update totals
        total_control_visitors += batch_size
        total_treatment_visitors += batch_size
        total_control_conversions += new_control_conversions
        total_treatment_conversions += new_treatment_conversions
        
        # Analyze current state
        analyzer = ABTestAnalyzer(
            total_control_conversions, total_control_visitors,
            total_treatment_conversions, total_treatment_visitors
        )
        
        summary = analyzer.get_summary()
        summary['batch'] = batch
        summary['total_visitors'] = total_control_visitors + total_treatment_visitors
        
        results.append(summary)
    
    return pd.DataFrame(results)

def plot_sequential_results(results_df):
    """
    Plot sequential testing results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # P-value over time
    axes[0, 0].plot(results_df['batch'], results_df['p_value'], marker='o')
    axes[0, 0].axhline(0.05, color='red', linestyle='--', label='α = 0.05')
    axes[0, 0].set_xlabel('Batch Number')
    axes[0, 0].set_ylabel('P-value')
    axes[0, 0].set_title('P-value Evolution')
    axes[0, 0].legend()
    
    # Lift over time
    axes[0, 1].plot(results_df['batch'], results_df['lift'], marker='o', color='green')
    axes[0, 1].set_xlabel('Batch Number')
    axes[0, 1].set_ylabel('Lift')
    axes[0, 1].set_title('Observed Lift Over Time')
    
    # Conversion rates
    axes[1, 0].plot(results_df['batch'], results_df['control_rate'], 
                    marker='o', label='Control', color='blue')
    axes[1, 0].plot(results_df['batch'], results_df['treatment_rate'], 
                    marker='o', label='Treatment', color='orange')
    axes[1, 0].set_xlabel('Batch Number')
    axes[1, 0].set_ylabel('Conversion Rate')
    axes[1, 0].set_title('Conversion Rates Over Time')
    axes[1, 0].legend()
    
    # Effect size
    axes[1, 1].plot(results_df['batch'], results_df['effect_size'], 
                    marker='o', color='purple')
    axes[1, 1].set_xlabel('Batch Number')
    axes[1, 1].set_ylabel('Effect Size (Cohen\'s d)')
    axes[1, 1].set_title('Effect Size Over Time')
    
    plt.tight_layout()
    return fig
