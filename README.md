# Website-AB-Testing-
This is an A/B Testing Dashboard and Analysis Tool built with Python. It's designed to help businesses and data analysts run, simulate, and analyze A/B tests (also called split tests) for website conversion optimization and other experimental scenarios.

## Understanding A/B Testing
A/B testing is a statistical method used to compare two versions of something (like a webpage, email, or app feature) to determine which performs better. Here's how it works:

### Version A (Control): The current version

### Version B (Treatment): The new version you want to test

### Goal: Determine if Version B significantly improves a key metric (like conversion rate, click-through rate, etc.)

## Why A/B Testing Matters
Data-driven decisions: Make changes based on actual user behavior, not assumptions

Risk reduction: Test changes on a small group before full rollout

Continuous optimization: Systematically improve your product or website

ROI measurement: Quantify the impact of changes

## Project Components
### 1. Main Dashboard (app.py)
This is a Streamlit web application that provides an interactive interface for:

Setting up A/B test parameters

Running test simulations

Visualizing results with charts and graphs

Analyzing statistical significance

Generating comprehensive reports

Key Features:

User-friendly web interface

Real-time data visualization

Interactive parameter adjustment

Professional styling and layout

### 2. Analysis Engine (ab_testing.py)
This is the core analytical library containing the ABTestAnalyzer class with advanced statistical capabilities:

Statistical Analysis Methods:
Proportion confidence intervals: Calculate uncertainty ranges for conversion rates

Two-proportion z-test: Determine if differences between groups are statistically significant

Effect size calculation: Measure the practical significance using Cohen's d

Power analysis: Estimate required sample sizes for reliable results

Simulation Capabilities:
Basic A/B test simulation: Generate synthetic test data for learning and validation

Sequential testing: Simulate how results evolve over time as more data comes in

Batch analysis: Analyze tests that run in phases or time periods
