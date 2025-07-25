import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from IPython.display import clear_output

# Set page configuration
st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧪 A/B Testing Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        This interactive dashboard allows you to simulate and analyze A/B testing results 
        for website conversion optimization.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("📋 Test Configuration")

# Input parameters
n_visitors_A = st.sidebar.number_input(
    "Visitors for Variant A", 
    min_value=100, 
    max_value=100000, 
    value=10000, 
    step=100,
    help="Number of visitors exposed to Variant A"
)

n_visitors_B = st.sidebar.number_input(
    "Visitors for Variant B", 
    min_value=100, 
    max_value=100000, 
    value=10000, 
    step=100,
    help="Number of visitors exposed to Variant B"
)

true_rate_A = st.sidebar.slider(
    "True Conversion Rate A (%)", 
    min_value=1.0, 
    max_value=50.0, 
    value=10.0, 
    step=0.1,
    help="Actual conversion rate for Variant A"
) / 100

true_rate_B = st.sidebar.slider(
    "True Conversion Rate B (%)", 
    min_value=1.0, 
    max_value=50.0, 
    value=12.0, 
    step=0.1,
    help="Actual conversion rate for Variant B"
) / 100

confidence_level = st.sidebar.selectbox(
    "Confidence Level", 
    [90, 95, 99], 
    index=1,
    help="Statistical confidence level for the test"
)
alpha = (100 - confidence_level) / 100

# Advanced settings in expandable section
with st.sidebar.expander("⚙️ Advanced Settings"):
    seed = st.number_input("Random Seed", value=42, help="For reproducible results")
    show_sequential = st.checkbox("Enable Sequential Testing", value=True)
    if show_sequential:
        batch_size = st.number_input("Batch Size", min_value=50, max_value=1000, value=100)
        n_batches = st.number_input("Number of Batches", min_value=10, max_value=100, value=60)

# Utility functions
def proportion_ci(successes, n, alpha=0.05):
    """Calculate confidence interval for proportion"""
    p_hat = successes / n
    z = stats.norm.ppf(1 - alpha/2)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    return p_hat, p_hat - z*se, p_hat + z*se

def calculate_effect_size(p1, p2, n1, n2):
    """Calculate Cohen's d for proportions"""
    pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
    pooled_se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
    return (p2 - p1) / pooled_se if pooled_se > 0 else 0

def power_analysis(effect_size, alpha=0.05, power=0.8):
    """Simple power analysis estimation"""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2 if effect_size > 0 else float('inf')
    return n

# Main application
if st.sidebar.button("🚀 Run A/B Test Simulation", type="primary"):
    
    # Set random seed
    np.random.seed(seed)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate data
    status_text.text("Simulating visitor data...")
    progress_bar.progress(25)
    
    conversions_A = np.random.binomial(n_visitors_A, true_rate_A)
    conversions_B = np.random.binomial(n_visitors_B, true_rate_B)
    
    # Calculate observed rates
    observed_rate_A = conversions_A / n_visitors_A
    observed_rate_B = conversions_B / n_visitors_B
    
    progress_bar.progress(50)
    status_text.text("Calculating confidence intervals...")
    
    # Calculate confidence intervals
    rate_A, ci_low_A, ci_high_A = proportion_ci(conversions_A, n_visitors_A, alpha)
    rate_B, ci_low_B, ci_high_B = proportion_ci(conversions_B, n_visitors_B, alpha)
    
    progress_bar.progress(75)
    status_text.text("Performing statistical tests...")
    
    # Statistical test
    count = np.array([conversions_B, conversions_A])
    nobs = np.array([n_visitors_B, n_visitors_A])
    z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')
    
    # Calculate additional metrics
    lift = observed_rate_B - observed_rate_A
    lift_percent = (lift / observed_rate_A) * 100 if observed_rate_A > 0 else 0
    effect_size = calculate_effect_size(observed_rate_A, observed_rate_B, n_visitors_A, n_visitors_B)
    
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    time.sleep(0.5)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    st.header("📊 Test Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🅰️ Variant A Conversion Rate",
            value=f"{observed_rate_A:.2%}",
            delta=f"{conversions_A} conversions"
        )
    
    with col2:
        st.metric(
            label="🅱️ Variant B Conversion Rate", 
            value=f"{observed_rate_B:.2%}",
            delta=f"{conversions_B} conversions"
        )
    
    with col3:
        st.metric(
            label="📈 Absolute Lift",
            value=f"{lift:.2%}",
            delta=f"{lift_percent:+.1f}% relative"
        )
    
    with col4:
        st.metric(
            label="📏 Effect Size (Cohen's d)",
            value=f"{effect_size:.3f}",
            delta="Small" if abs(effect_size) < 0.2 else "Medium" if abs(effect_size) < 0.8 else "Large"
        )
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    results_df = pd.DataFrame({
        'Variant': ['A (Control)', 'B (Treatment)'],
        'Visitors': [f"{n_visitors_A:,}", f"{n_visitors_B:,}"],
        'Conversions': [f"{conversions_A:,}", f"{conversions_B:,}"],
        'Conversion Rate': [f"{observed_rate_A:.3%}", f"{observed_rate_B:.3%}"],
        'Lower CI': [f"{ci_low_A:.3%}", f"{ci_low_B:.3%}"],
        'Upper CI': [f"{ci_high_A:.3%}", f"{ci_high_B:.3%}"]
    })
    
    st.dataframe(results_df, use_container_width=True)
    
    # Visualizations
    st.subheader("📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Conversion rates with confidence intervals
        fig = go.Figure()
        
        variants = ['Variant A', 'Variant B']
        rates = [observed_rate_A, observed_rate_B]
        colors = ['#1f77b4', '#ff7f0e']
        
        for i, (variant, rate, color) in enumerate(zip(variants, rates, colors)):
            ci_low = ci_low_A if i == 0 else ci_low_B
            ci_high = ci_high_A if i == 0 else ci_high_B
            
            fig.add_trace(go.Bar(
                x=[variant],
                y=[rate],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[ci_high - rate],
                    arrayminus=[rate - ci_low]
                ),
                marker_color=color,
                name=variant
            ))
        
        fig.update_layout(
            title=f'Conversion Rates with {confidence_level}% Confidence Intervals',
            yaxis_title='Conversion Rate',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Statistical test results
        st.markdown("### 🧮 Statistical Test Results")
        
        is_significant = p_value < alpha
        
        # Create a nice results box
        if is_significant:
            st.markdown(f"""
            <div class="success-message">
                <h4>✅ STATISTICALLY SIGNIFICANT</h4>
                <p><strong>Z-statistic:</strong> {z_stat:.3f}</p>
                <p><strong>P-value:</strong> {p_value:.4f}</p>
                <p><strong>Significance Level:</strong> {alpha:.3f}</p>
                <p><strong>Conclusion:</strong> Variant B has a significantly higher conversion rate than Variant A!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-message">
                <h4>❌ NOT STATISTICALLY SIGNIFICANT</h4>
                <p><strong>Z-statistic:</strong> {z_stat:.3f}</p>
                <p><strong>P-value:</strong> {p_value:.4f}</p>
                <p><strong>Significance Level:</strong> {alpha:.3f}</p>
                <p><strong>Conclusion:</strong> No significant difference detected between variants.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Power analysis
        required_sample_size = power_analysis(effect_size, alpha, 0.8)
        st.markdown(f"""
        **Power Analysis:**
        - Required sample size per group (80% power): {required_sample_size:.0f}
        - Current sample size per group: {min(n_visitors_A, n_visitors_B):,}
        """)
    
    # Sequential testing
    if show_sequential:
        st.subheader("🔄 Sequential Testing Simulation")
        
        with st.expander("View Sequential Analysis", expanded=False):
            # Initialize counters
            n_visits_A = n_visits_B = 0
            n_success_A = n_success_B = 0
            
            # Lists to store metrics for plotting
            batches = []
            p_values = []
            observed_lifts = []
            conversion_rates_A = []
            conversion_rates_B = []
            
            # Create placeholder for real-time updates
            chart_placeholder = st.empty()
            
            for batch in range(1, min(n_batches + 1, 21)):  # Limit to 20 batches for performance
                # Simulate one batch of visitors
                new_A = np.random.binomial(batch_size, true_rate_A)
                new_B = np.random.binomial(batch_size, true_rate_B)
                
                # Update totals
                n_visits_A += batch_size
                n_visits_B += batch_size
                n_success_A += new_A
                n_success_B += new_B
                
                # Compute current conversion rates
                cr_A = n_success_A / n_visits_A
                cr_B = n_success_B / n_visits_B
                lift = cr_B - cr_A
                
                # Two proportion z-test
                count = np.array([n_success_B, n_success_A])
                nobs = np.array([n_visits_B, n_visits_A])
                z_test, p_val = proportions_ztest(count, nobs, alternative='larger')
                
                # Store metrics
                batches.append(batch)
                p_values.append(p_val)
                observed_lifts.append(lift)
                conversion_rates_A.append(cr_A)
                conversion_rates_B.append(cr_B)
            
            # Create sequential testing visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('P-value Over Time', 'Observed Lift Over Time', 
                              'Conversion Rates Over Time', 'Statistical Power'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # P-value plot
            fig.add_trace(
                go.Scatter(x=batches, y=p_values, mode='lines+markers', name='P-value', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_hline(y=alpha, line_dash="dash", line_color="red", row=1, col=1)
            
            # Lift plot
            fig.add_trace(
                go.Scatter(x=batches, y=observed_lifts, mode='lines+markers', name='Lift', line=dict(color='green')),
                row=1, col=2
            )
            
            # Conversion rates plot
            fig.add_trace(
                go.Scatter(x=batches, y=conversion_rates_A, mode='lines+markers', name='Variant A', line=dict(color='#1f77b4')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=batches, y=conversion_rates_B, mode='lines+markers', name='Variant B', line=dict(color='#ff7f0e')),
                row=2, col=1
            )
            
            # Sample size evolution
            sample_sizes = [i * batch_size for i in batches]
            fig.add_trace(
                go.Scatter(x=batches, y=sample_sizes, mode='lines+markers', name='Sample Size', line=dict(color='purple')),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True, title_text="Sequential A/B Testing Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Final sequential results
            final_significant = p_values[-1] < alpha if p_values else False
            
            st.markdown(f"""
            **Sequential Testing Summary:**
            - Final sample size per variant: {n_visits_A:,}
            - Final p-value: {p_values[-1]:.4f}
            - Final lift: {observed_lifts[-1]:.2%}
            - Result: {'✅ Significant' if final_significant else '❌ Not Significant'}
            """)

# Sample data generation section  
st.header("📋 Sample Scenarios")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎯 High Impact Test", help="Large effect size scenario"):
        st.session_state.scenario = "high_impact"
        st.info("Scenario: Control 5%, Treatment 8% (+60% relative lift)")

with col2:
    if st.button("🔍 Small Effect Test", help="Small effect size scenario"):
        st.session_state.scenario = "small_effect"
        st.info("Scenario: Control 10%, Treatment 10.5% (+5% relative lift)")

with col3:
    if st.button("⚖️ No Effect Test", help="No difference scenario"):
        st.session_state.scenario = "no_effect"
        st.info("Scenario: Control 8%, Treatment 8% (no difference)")

# Information section
with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown("""
    ### Getting Started
    1. **Configure Test Parameters**: Use the sidebar to set visitor counts and conversion rates
    2. **Run Simulation**: Click the "Run A/B Test Simulation" button
    3. **Analyze Results**: Review the metrics, confidence intervals, and statistical significance
    4. **Interpret Results**: 
       - Green metrics indicate positive results
       - Check if p-value < significance level for statistical significance
       - Review confidence intervals for practical significance
       - Consider effect size for business impact
    
    ### Understanding the Results
    - **Conversion Rate**: Percentage of visitors who completed the desired action
    - **Confidence Interval**: Range where we expect the true conversion rate to fall
    - **P-value**: Probability that the observed difference occurred by chance
    - **Effect Size**: Magnitude of the difference (Cohen's d)
    - **Lift**: Absolute and relative improvement from variant A to B
    
    ### Sequential Testing
    - Simulates how results evolve as more data is collected
    - Helps understand when you might stop a test early
    - Shows the stability of your results over time
    
    ### Best Practices
    - Run tests for full business cycles (typically 1-2 weeks minimum)
    - Ensure adequate sample sizes for reliable results
    - Consider practical significance alongside statistical significance
    - Account for multiple testing if running many experiments
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>For questions or support, please refer to the documentation or open an issue on GitHub.</p>
</div>
""", unsafe_allow_html=True)
