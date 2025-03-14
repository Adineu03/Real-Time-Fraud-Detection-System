import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import time
from utils.data_processor import process_uploaded_file, filter_dataframe
from utils.risk_models import (
    calculate_bayesian_score,
    calculate_mle_score,
    calculate_combined_score,
    calculate_fuzzy_score,
    flag_transactions
)
from utils.visualization import (
    create_trend_chart,
    create_distribution_chart,
    create_roc_curve,
    create_risk_breakdown
)

# Set page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing data and variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'transaction_count' not in st.session_state:
    st.session_state.transaction_count = 0
if 'flagged_count' not in st.session_state:
    st.session_state.flagged_count = 0
if 'selected_transaction' not in st.session_state:
    st.session_state.selected_transaction = None
if 'timestamp' not in st.session_state:
    st.session_state.timestamp = time.time()

# Function to reset session state when new file is uploaded
def reset_session_state():
    st.session_state.filtered_data = None
    st.session_state.transaction_count = 0
    st.session_state.flagged_count = 0
    st.session_state.selected_transaction = None
    st.session_state.timestamp = time.time()

# Header
st.title("🔍 Real-Time Fraud Detection Dashboard")
st.markdown("Upload transaction data, analyze risk scores, and identify potential fraud.")

# Sidebar for file upload and filters
with st.sidebar:
    st.header("Data Upload & Filters")
    
    # File upload section
    st.subheader("Upload Transaction Data")
    st.markdown("Upload CSV or Excel files containing transaction data.")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            df = process_uploaded_file(uploaded_file)
            
            # Reset session state for new data
            reset_session_state()
            
            # Calculate risk scores
            df['bayesian_score'] = calculate_bayesian_score(df)
            df['mle_score'] = calculate_mle_score(df)
            df['combined_score'] = calculate_combined_score(df['bayesian_score'], df['mle_score'])
            df['fuzzy_score'] = calculate_fuzzy_score(df['combined_score'], df)
            
            # Flag transactions based on risk scores
            df = flag_transactions(df)
            
            # Update session state
            st.session_state.data = df
            st.session_state.filtered_data = df
            st.session_state.transaction_count = len(df)
            st.session_state.flagged_count = df['is_flagged'].sum()
            
            st.success(f"Successfully processed {len(df)} transactions")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Only show filters if data is loaded
    if st.session_state.data is not None:
        st.subheader("Filter Transactions")
        
        # Date filter if date column exists
        if 'date' in st.session_state.data.columns:
            min_date = pd.to_datetime(st.session_state.data['date']).min().date()
            max_date = pd.to_datetime(st.session_state.data['date']).max().date()
            date_range = st.date_input(
                "Date Range",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
        
        # Amount filter if amount column exists
        if 'amount' in st.session_state.data.columns:
            min_amount = float(st.session_state.data['amount'].min())
            max_amount = float(st.session_state.data['amount'].max())
            amount_range = st.slider(
                "Amount Range",
                min_amount,
                max_amount,
                (min_amount, max_amount)
            )
        
        # Risk score filter
        risk_threshold = st.slider(
            "Minimum Risk Score",
            0.0,
            1.0,
            0.7
        )
        
        # Filter options
        filter_options = st.multiselect(
            "Show only:",
            ["All Transactions", "Flagged Transactions"],
            default=["All Transactions"]
        )
        
        # Apply filters button
        if st.button("Apply Filters"):
            # Build filter conditions
            filter_conditions = {}
            
            if 'date' in st.session_state.data.columns and len(date_range) == 2:
                filter_conditions['date'] = (date_range[0], date_range[1])
            
            if 'amount' in st.session_state.data.columns:
                filter_conditions['amount'] = amount_range
            
            filter_conditions['risk_threshold'] = risk_threshold
            
            # Handle show only options
            show_flagged_only = "Flagged Transactions" in filter_options and "All Transactions" not in filter_options
            
            # Apply filters
            st.session_state.filtered_data = filter_dataframe(
                st.session_state.data,
                filter_conditions,
                show_flagged_only
            )
            
            # Update counts
            st.session_state.transaction_count = len(st.session_state.filtered_data)
            st.session_state.flagged_count = st.session_state.filtered_data['is_flagged'].sum()
            
            st.success(f"Filters applied. Showing {len(st.session_state.filtered_data)} transactions.")

# Main content area
if st.session_state.data is not None:
    # Layout with multiple columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", st.session_state.transaction_count)
    
    with col2:
        st.metric("Flagged Transactions", st.session_state.flagged_count)
    
    with col3:
        flagged_percentage = (st.session_state.flagged_count / st.session_state.transaction_count * 100) if st.session_state.transaction_count > 0 else 0
        st.metric("Flagged Percentage", f"{flagged_percentage:.2f}%")
    
    with col4:
        avg_risk = st.session_state.filtered_data['combined_score'].mean() if st.session_state.filtered_data is not None else 0
        st.metric("Average Risk Score", f"{avg_risk:.4f}")
    
    # Transaction table with color coding
    st.subheader("Transaction Data")
    
    # Search functionality
    search_term = st.text_input("Search Transactions", "")
    
    # Filter data based on search term
    if search_term:
        search_results = st.session_state.filtered_data[
            st.session_state.filtered_data.astype(str).apply(
                lambda row: row.str.contains(search_term, case=False).any(), 
                axis=1
            )
        ]
    else:
        search_results = st.session_state.filtered_data
    
    # Style the dataframe
    def color_risk_scores(val):
        if isinstance(val, (int, float)):
            if val >= 0.8:
                return 'background-color: #ffcccc'  # Light red
            elif val >= 0.5:
                return 'background-color: #ffffcc'  # Light yellow
            elif val >= 0:
                return 'background-color: #ccffcc'  # Light green
        return ''
    
    # Function to highlight flagged rows
    def highlight_flagged(row):
        if row['is_flagged']:
            return ['background-color: #ffe6e6'] * len(row)
        return [''] * len(row)
    
    # Display dataframe with styling
    if not search_results.empty:
        # Format the DataFrame for display
        display_columns = search_results.columns.tolist()
        
        # Ensure risk scores appear in preferred order and with proper formatting
        score_columns = ['bayesian_score', 'mle_score', 'combined_score', 'fuzzy_score']
        for col in score_columns:
            if col in display_columns:
                search_results[col] = search_results[col].round(4)
        
        # Clickable rows for detailed view
        selected_indices = st.dataframe(
            search_results.style
            .applymap(color_risk_scores, subset=score_columns)
            .apply(highlight_flagged, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Check if a row is selected
        if st.button("View Details of Selected Transaction"):
            try:
                if selected_indices is not None and len(selected_indices) > 0:
                    selected_idx = selected_indices.index[0]
                    st.session_state.selected_transaction = search_results.iloc[selected_idx]
                else:
                    st.warning("Please select a transaction row first.")
            except Exception as e:
                st.warning("Please select a transaction row first.")
    else:
        st.info("No transactions match the current filters or search criteria.")
    
    # Charts and visualizations - 2 column layout
    st.subheader("Transaction Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Score Distribution")
        risk_dist_chart = create_distribution_chart(st.session_state.filtered_data)
        st.plotly_chart(risk_dist_chart, use_container_width=True)
    
    with col2:
        st.markdown("### Risk Score Trend")
        if 'date' in st.session_state.filtered_data.columns:
            trend_chart = create_trend_chart(st.session_state.filtered_data)
            st.plotly_chart(trend_chart, use_container_width=True)
        else:
            st.info("Date information not available for trend analysis.")
    
    # Performance metrics
    st.subheader("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        roc_curve = create_roc_curve()
        st.plotly_chart(roc_curve, use_container_width=True)
    
    with col2:
        # Model comparison bar chart
        model_comparison = go.Figure()
        model_comparison.add_trace(go.Bar(
            x=['Bayesian', 'MLE', 'Combined', 'Fuzzy'],
            y=[0.82, 0.78, 0.85, 0.89],  # Example precision values
            name='Precision'
        ))
        model_comparison.add_trace(go.Bar(
            x=['Bayesian', 'MLE', 'Combined', 'Fuzzy'],
            y=[0.75, 0.81, 0.83, 0.84],  # Example recall values
            name='Recall'
        ))
        model_comparison.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            legend_title='Metric',
            barmode='group'
        )
        st.plotly_chart(model_comparison, use_container_width=True)
    
    # Transaction Details Section (if a transaction is selected)
    if st.session_state.selected_transaction is not None:
        st.subheader("Transaction Details")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Transaction Information")
            for col, val in st.session_state.selected_transaction.items():
                if col not in ['bayesian_score', 'mle_score', 'combined_score', 'fuzzy_score', 'is_flagged']:
                    st.text(f"{col}: {val}")
            
            # Risk scores with color coding
            st.markdown("### Risk Scores")
            
            bayesian = st.session_state.selected_transaction['bayesian_score']
            mle = st.session_state.selected_transaction['mle_score']
            combined = st.session_state.selected_transaction['combined_score']
            fuzzy = st.session_state.selected_transaction['fuzzy_score']
            
            # Function to get color based on score
            def get_color(score):
                if score >= 0.8:
                    return "red"
                elif score >= 0.5:
                    return "orange"
                else:
                    return "green"
            
            # Display scores with color coding
            st.markdown(f"**Bayesian Score:** <span style='color:{get_color(bayesian)}'>{bayesian:.4f}</span>", unsafe_allow_html=True)
            st.markdown(f"**MLE Score:** <span style='color:{get_color(mle)}'>{mle:.4f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Combined Score:** <span style='color:{get_color(combined)}'>{combined:.4f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Fuzzy Score:** <span style='color:{get_color(fuzzy)}'>{fuzzy:.4f}</span>", unsafe_allow_html=True)
            
            # Flagged status
            is_flagged = st.session_state.selected_transaction['is_flagged']
            if is_flagged:
                st.markdown("**Status:** <span style='color:red'>⚠️ FLAGGED</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Status:** <span style='color:green'>✓ Normal</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Risk Breakdown")
            risk_breakdown = create_risk_breakdown(st.session_state.selected_transaction)
            st.plotly_chart(risk_breakdown, use_container_width=True)
            
            st.markdown("### Contributing Factors")
            # Example contributing factors based on transaction data
            # In a real system, these would be determined by the model
            
            # Generate some example factors
            factors = []
            
            if 'amount' in st.session_state.selected_transaction:
                amount = st.session_state.selected_transaction['amount']
                if amount > 1000:
                    factors.append(("High transaction amount", 0.35))
            
            if 'location' in st.session_state.selected_transaction:
                factors.append(("Unusual location", 0.25))
            
            if 'merchant' in st.session_state.selected_transaction:
                factors.append(("Merchant risk category", 0.15))
            
            if 'time' in st.session_state.selected_transaction:
                factors.append(("Transaction time anomaly", 0.10))
            
            # Add a generic factor if nothing specific was found
            if not factors:
                factors = [
                    ("Transaction pattern anomaly", 0.30),
                    ("User behavior deviation", 0.25),
                    ("Geographical risk", 0.20),
                    ("Merchant category risk", 0.15),
                    ("Transaction timing", 0.10)
                ]
            
            # Display the factors
            for factor, contribution in factors:
                st.markdown(f"- **{factor}**: {contribution:.2f} contribution weight")
            
            st.markdown("### Recommended Actions")
            st.markdown("""
            - Review transaction details with customer
            - Request additional verification if necessary
            - Flag account for monitoring
            - Document decision in the system
            """)
    
    # Real-time alerts section
    st.subheader("Real-Time Alerts")
    
    # Simulate real-time alerts by showing recent flagged transactions
    if st.session_state.filtered_data is not None:
        flagged_transactions = st.session_state.filtered_data[st.session_state.filtered_data['is_flagged'] == True]
        
        if not flagged_transactions.empty:
            st.warning(f"⚠️ {len(flagged_transactions)} transactions have been flagged as potentially fraudulent.")
            
            # Show the most recent 3 flagged transactions
            st.markdown("### Recent Flagged Transactions")
            for idx, row in flagged_transactions.head(3).iterrows():
                with st.expander(f"Transaction ID: {row.get('transaction_id', idx)}"):
                    for col, val in row.items():
                        if col in ['bayesian_score', 'mle_score', 'combined_score', 'fuzzy_score']:
                            st.markdown(f"**{col}:** {val:.4f}")
                        elif col == 'is_flagged':
                            continue
                        else:
                            st.markdown(f"**{col}:** {val}")
        else:
            st.success("No fraudulent transactions detected in the current dataset.")

else:
    # Display upload instructions when no data is loaded
    st.info("👈 Please upload a transaction file (CSV or Excel) using the sidebar to start analyzing fraud data.")
    
    # Display example of what the dashboard will show
    st.subheader("Dashboard Preview")
    st.markdown("""
    This dashboard will help you:
    
    1. **Analyze transaction data** with advanced risk scoring models
    2. **Identify potentially fraudulent transactions** with multiple detection methods
    3. **Visualize trends and patterns** in your transaction data
    4. **Get detailed insights** into flagged transactions
    
    Upload your transaction data to get started.
    """)
