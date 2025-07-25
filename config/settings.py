"""Application configuration settings"""

APP_CONFIG = {
    'app_name': 'Enhanced A/B Testing Dashboard',
    'version': '2.0.0',
    'author': 'Your Name',
    'description': 'Advanced statistical analysis for A/B testing',
    
    # Default test parameters
    'defaults': {
        'visitors_per_variant': 10000,
        'baseline_conversion_rate': 0.10,
        'minimum_detectable_effect': 0.02,
        'confidence_level': 95,
        'statistical_power': 0.80,
        'alpha': 0.05
    },
    
    # Visualization settings
    'charts': {
        'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'default_height': 400,
        'dpi': 300
    },
    
    # Export settings
    'export': {
        'max_file_size_mb': 50,
        'supported_formats': ['csv', 'json', 'pdf', 'html'],
        'timestamp_format': '%Y%m%d_%H%M%S'
    },
    
    # Statistical settings
    'statistics': {
        'minimum_sample_size': 100,
        'maximum_sample_size': 1000000,
        'bootstrap_iterations': 1000,
        'monte_carlo_simulations': 10000
    }
}

# Email configuration (for notifications)
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'use_tls': True,
    'sender_email': '',  # Configure with your email
    'sender_password': '',  # Configure with app password
}

# Database configuration (for test history)
DATABASE_CONFIG = {
    'type': 'sqlite',  # or 'postgresql', 'mysql'
    'filename': 'ab_test_history.db',
    'table_name': 'test_results'
}
