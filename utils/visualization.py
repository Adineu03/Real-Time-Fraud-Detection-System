import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import random

def create_trend_chart(df):
    """
    Create a line chart showing risk score trends over time
    
    Parameters:
    df (pd.DataFrame): Transaction data with risk scores and date information
    
    Returns:
    go.Figure: Plotly figure with trend chart
    """
    # Ensure we have date information
    if 'date' not in df.columns:
        # Create empty chart with message if no date column
        fig = go.Figure()
        fig.add_annotation(
            text="Date information not available for trend analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Aggregate data by date
    daily_data = df.groupby(df['date'].dt.date).agg({
        'bayesian_score': 'mean',
        'mle_score': 'mean',
        'combined_score': 'mean',
        'fuzzy_score': 'mean',
        'is_flagged': 'sum'
    }).reset_index()
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each score type
    fig.add_trace(go.Scatter(
        x=daily_data['date'],
        y=daily_data['bayesian_score'],
        mode='lines',
        name='Bayesian Score',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_data['date'],
        y=daily_data['mle_score'],
        mode='lines',
        name='MLE Score',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_data['date'],
        y=daily_data['combined_score'],
        mode='lines',
        name='Combined Score',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_data['date'],
        y=daily_data['fuzzy_score'],
        mode='lines',
        name='Fuzzy Score',
        line=dict(color='red')
    ))
    
    # Add flagged transactions count as bar chart on secondary y-axis
    fig.add_trace(go.Bar(
        x=daily_data['date'],
        y=daily_data['is_flagged'],
        name='Flagged Count',
        marker_color='rgba(200, 0, 0, 0.3)',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Risk Score Trends Over Time',
        xaxis_title='Date',
        yaxis_title='Risk Score',
        yaxis2=dict(
            title='Flagged Count',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_distribution_chart(df):
    """
    Create a histogram showing the distribution of risk scores
    
    Parameters:
    df (pd.DataFrame): Transaction data with risk scores
    
    Returns:
    go.Figure: Plotly figure with distribution chart
    """
    # Create figure
    fig = go.Figure()
    
    # Add histograms for each score type
    fig.add_trace(go.Histogram(
        x=df['fuzzy_score'],
        name='Fuzzy Score',
        marker_color='red',
        opacity=0.5,
        xbins=dict(
            start=0,
            end=1,
            size=0.05
        )
    ))
    
    fig.add_trace(go.Histogram(
        x=df['combined_score'],
        name='Combined Score',
        marker_color='orange',
        opacity=0.5,
        xbins=dict(
            start=0,
            end=1,
            size=0.05
        )
    ))
    
    # Add vertical line for typical threshold
    fig.add_shape(
        type="line",
        x0=0.8, y0=0, 
        x1=0.8, y1=1,
        yref="paper",
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        )
    )
    
    # Add annotation for threshold line
    fig.add_annotation(
        x=0.8,
        y=1,
        yref="paper",
        text="Threshold",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-30
    )
    
    # Update layout
    fig.update_layout(
        title='Risk Score Distribution',
        xaxis_title='Risk Score',
        yaxis_title='Number of Transactions',
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_roc_curve():
    """
    Create a ROC curve chart for model performance
    
    Note: In a real system, this would use actual model predictions and ground truth.
    For this demo, we generate synthetic ROC curves.
    
    Returns:
    go.Figure: Plotly figure with ROC curve
    """
    # Create synthetic ROC curve data for demonstration
    # In a real application, this would use actual model performance data
    
    # Generate random but realistic looking ROC curves for each model
    np.random.seed(42)  # For reproducibility
    
    # Function to generate synthetic ROC curve data
    def generate_roc_data(auc_target):
        # Generate points that approximate a ROC curve with the given AUC
        x = np.linspace(0, 1, 100)
        # Create a curve that roughly matches the target AUC
        y = x ** ((1 - auc_target) * 2)
        # Add some noise
        noise = np.random.normal(0, 0.03, size=len(x))
        y = np.clip(y + noise, 0, 1)
        # Sort to ensure curve is monotonic
        idx = np.argsort(x)
        return x[idx], y[idx]
    
    # Generate ROC curves with target AUCs
    fpr_bayes, tpr_bayes = generate_roc_data(0.82)
    fpr_mle, tpr_mle = generate_roc_data(0.78)
    fpr_combined, tpr_combined = generate_roc_data(0.85)
    fpr_fuzzy, tpr_fuzzy = generate_roc_data(0.89)
    
    # Create figure
    fig = go.Figure()
    
    # Add diagonal reference line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1], 
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash'),
        showlegend=True
    ))
    
    # Add ROC curves for each model
    fig.add_trace(go.Scatter(
        x=fpr_bayes, 
        y=tpr_bayes,
        mode='lines',
        name=f'Bayesian (AUC = 0.82)',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=fpr_mle, 
        y=tpr_mle,
        mode='lines',
        name=f'MLE (AUC = 0.78)',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=fpr_combined, 
        y=tpr_combined,
        mode='lines',
        name=f'Combined (AUC = 0.85)',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=fpr_fuzzy, 
        y=tpr_fuzzy,
        mode='lines',
        name=f'Fuzzy (AUC = 0.89)',
        line=dict(color='red')
    ))
    
    # Update layout
    fig.update_layout(
        title='ROC Curve - Model Performance',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=600,
        height=500
    )
    
    # Update axes to be between 0 and 1
    fig.update_xaxes(range=[0, 1], constrain='domain')
    fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
    
    return fig

def create_risk_breakdown(transaction):
    """
    Create a radar chart showing breakdown of risk factors for a transaction
    
    Parameters:
    transaction (pd.Series): Single transaction data
    
    Returns:
    go.Figure: Plotly figure with risk factor breakdown
    """
    # In a real system, these would be actual risk factors from the model
    # For this demo, we'll create synthetic risk factors
    
    # Define risk categories
    categories = [
        'Transaction Amount', 
        'Transaction Timing',
        'Location Risk',
        'Merchant Category',
        'User History'
    ]
    
    # Create synthetic risk values
    # In a real system, these would come from the actual model
    risk_values = []
    
    # Amount risk (if amount exists)
    if 'amount' in transaction:
        # Higher amounts = higher risk
        amount = transaction['amount']
        if amount > 1000:
            risk_values.append(0.9)
        elif amount > 500:
            risk_values.append(0.7)
        elif amount > 100:
            risk_values.append(0.4)
        else:
            risk_values.append(0.2)
    else:
        risk_values.append(random.uniform(0.3, 0.8))
    
    # Time risk (if time or date exists)
    time_risk = 0.5  # Default
    if 'time' in transaction:
        # Extract hour if possible
        try:
            hour = int(str(transaction['time']).split(':')[0])
            if 1 <= hour <= 5:  # Late night/early morning
                time_risk = 0.9
            elif 9 <= hour <= 17:  # Business hours
                time_risk = 0.3
            else:  # Evening
                time_risk = 0.6
        except:
            pass
    elif 'date' in transaction and hasattr(transaction['date'], 'hour'):
        # If date is a datetime object with hour attribute
        hour = transaction['date'].hour
        if 1 <= hour <= 5:
            time_risk = 0.9
        elif 9 <= hour <= 17:
            time_risk = 0.3
        else:
            time_risk = 0.6
    
    risk_values.append(time_risk)
    
    # Location risk (if location exists)
    if 'location' in transaction:
        # Generate a consistent risk value based on location string
        location_hash = hash(str(transaction['location'])) % 100
        location_risk = location_hash / 100
        risk_values.append(location_risk)
    else:
        risk_values.append(random.uniform(0.3, 0.8))
    
    # Merchant category risk (if merchant exists)
    if 'merchant' in transaction or 'merchant_category' in transaction:
        merchant_field = 'merchant' if 'merchant' in transaction else 'merchant_category'
        # Generate a consistent risk value based on merchant string
        merchant_hash = hash(str(transaction[merchant_field])) % 100
        merchant_risk = merchant_hash / 100
        risk_values.append(merchant_risk)
    else:
        risk_values.append(random.uniform(0.3, 0.8))
    
    # User history risk (simulate)
    user_risk = random.uniform(0.2, 0.9)
    risk_values.append(user_risk)
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=risk_values,
        theta=categories,
        fill='toself',
        name='Risk Factors',
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title='Risk Factor Breakdown'
    )
    
    return fig
