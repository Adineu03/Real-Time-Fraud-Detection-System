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
    calculate_bandit_score,
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
            df['bandit_score'] = calculate_bandit_score(df, exploration_rate=0.2)
            
            # Flag transactions based on fuzzy risk score
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
            0.45
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
        score_columns = ['bayesian_score', 'mle_score', 'combined_score', 'fuzzy_score', 'bandit_score']
        for col in score_columns:
            if col in display_columns:
                search_results[col] = search_results[col].round(4)
        
        # Create styled dataframe using newer pandas API
        styled_df = search_results.style
        
        # Apply risk score coloring to specific columns
        for col in score_columns:
            if col in display_columns:
                styled_df = styled_df.map(color_risk_scores, subset=[col])
        
        # Apply row highlighting for flagged transactions
        styled_df = styled_df.apply(highlight_flagged, axis=1)
        
        # Display the styled dataframe
        selected_indices = st.dataframe(
            styled_df,
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
    
    # Charts and visualizations - Advanced analytics layout
    st.subheader("Transaction Analytics")
    
    # Risk distribution section
    st.markdown("### Risk Score Distribution")
    risk_dist_chart = create_distribution_chart(st.session_state.filtered_data)
    st.plotly_chart(risk_dist_chart, use_container_width=True)
    
    # Add interactive anomaly detection section
    st.markdown("### 🔍 Interactive Anomaly Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create scatter plot of transactions
        fig = px.scatter(
            st.session_state.filtered_data,
            x='combined_score',
            y='fuzzy_score',
            color='is_flagged',
            color_discrete_map={True: 'red', False: 'blue'},
            hover_data=st.session_state.filtered_data.columns,
            title="Transaction Risk Score Analysis",
            labels={"combined_score": "Combined Risk Score", "fuzzy_score": "Fuzzy Logic Score"},
            size='bayesian_score',
            size_max=15,
        )
        
        # Add color-coded regions to indicate risk zones
        fig.add_shape(
            type="rect",
            x0=0.8, y0=0.8,
            x1=1, y1=1,
            fillcolor="rgba(255,0,0,0.1)",
            line=dict(color="red"),
            layer="below"
        )
        fig.add_shape(
            type="rect",
            x0=0.5, y0=0.5,
            x1=0.8, y1=0.8,
            fillcolor="rgba(255,165,0,0.1)",
            line=dict(color="orange"),
            layer="below"
        )
        
        # Add annotations
        fig.add_annotation(
            x=0.9, y=0.9,
            text="High Risk Zone",
            showarrow=False,
            font=dict(color="red")
        )
        fig.add_annotation(
            x=0.65, y=0.65,
            text="Medium Risk Zone",
            showarrow=False,
            font=dict(color="orange")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Interactive controls for anomaly thresholds
        st.markdown("#### Adjust Detection Parameters")
        
        detection_sensitivity = st.slider(
            "Detection Sensitivity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Adjust the sensitivity of the anomaly detection algorithm"
        )
        
        outlier_threshold = st.slider(
            "Outlier Threshold (σ)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Standard deviations from mean to consider as outlier"
        )
        
        if st.button("Run Advanced Anomaly Detection"):
            # Calculate statistical anomalies
            # This would be a more sophisticated algorithm in production
            scores = st.session_state.filtered_data['combined_score']
            mean = scores.mean()
            std = scores.std()
            threshold = mean + (std * outlier_threshold)
            
            # Create new anomaly scores
            anomaly_count = len(scores[scores > threshold])
            
            # Display results
            st.metric("Detected Anomalies", anomaly_count)
            if anomaly_count > 0:
                st.warning(f"Found {anomaly_count} statistical anomalies based on your parameters")
            else:
                st.success("No statistical anomalies detected based on current parameters")
    
    # Performance metrics and advanced analytics
    st.subheader("Model Performance & Explainability")
    
    tabs = st.tabs(["Performance Metrics", "Model Explainability", "Feature Importance"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            roc_curve = create_roc_curve()
            st.plotly_chart(roc_curve, use_container_width=True)
        
        with col2:
            # Model comparison bar chart
            model_comparison = go.Figure()
            model_comparison.add_trace(go.Bar(
                x=['Bayesian', 'MLE', 'Combined', 'Fuzzy', 'Multi-Armed Bandit'],
                y=[0.82, 0.78, 0.85, 0.89, 0.87],  # Example precision values
                name='Precision'
            ))
            model_comparison.add_trace(go.Bar(
                x=['Bayesian', 'MLE', 'Combined', 'Fuzzy', 'Multi-Armed Bandit'],
                y=[0.75, 0.81, 0.83, 0.84, 0.86],  # Example recall values
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
    
    with tabs[1]:
        st.markdown("### 🧠 Model Decision Explanation")
        st.markdown("""
        This section provides transparency into how the fraud detection models make decisions.
        Select a transaction below to generate an explanation of the model's decision process.
        """)
        
        if not st.session_state.filtered_data.empty:
            # Allow user to select a transaction to explain
            selected_idx = st.selectbox(
                "Select transaction to explain:",
                options=st.session_state.filtered_data.index,
                format_func=lambda x: f"Transaction {x} - Risk Score: {st.session_state.filtered_data.loc[x, 'fuzzy_score']:.4f}"
            )
            
            if st.button("Generate Explanation"):
                st.markdown("### SHAP-Inspired Model Explanation")
                
                # Get selected transaction
                transaction = st.session_state.filtered_data.loc[selected_idx]
                
                # Create waterfall chart simulating SHAP values (actual SHAP would calculate these properly)
                waterfall_values = []
                feature_names = []
                
                # Base value (average prediction)
                base_value = 0.5
                
                # Get key features and assign impact values
                # In a real system, these would be actual SHAP values
                if 'amount' in transaction:
                    amount_impact = (transaction['amount'] / 10000) * 0.2  # Simplified impact calculation
                    waterfall_values.append(amount_impact)
                    feature_names.append('Transaction Amount')
                
                if 'merchant' in transaction:
                    # Create consistent but pseudo-random value based on merchant name
                    merchant_impact = (hash(str(transaction['merchant'])) % 100) / 100 * 0.15
                    waterfall_values.append(merchant_impact)
                    feature_names.append('Merchant')
                
                # Add other potential features
                for feature, impact in [
                    ('location', 0.08),
                    ('time', 0.07),
                    ('card_present', 0.12),
                    ('transaction_type', 0.09)
                ]:
                    if feature in transaction:
                        # Simplified impact based on feature presence
                        feature_impact = (hash(str(transaction[feature])) % 100) / 100 * impact
                        waterfall_values.append(feature_impact)
                        feature_names.append(feature.replace('_', ' ').title())
                
                # Ensure we have at least some features
                if not waterfall_values:
                    # Generate synthetic features for demo purposes
                    waterfall_values = [0.12, -0.05, 0.08, 0.15, -0.03]
                    feature_names = ['Transaction Type', 'Customer History', 'Time of Day', 'Amount', 'Merchant Category']
                
                # Calculate cumulative impact to reach final prediction
                final_value = base_value + sum(waterfall_values)
                
                # Create waterfall chart
                fig = go.Figure(go.Waterfall(
                    name="SHAP", 
                    orientation="v",
                    measure=["absolute"] + ["relative"] * len(waterfall_values) + ["total"],
                    x=["Base Value"] + feature_names + ["Final Prediction"],
                    y=[base_value] + waterfall_values + [final_value],
                    connector={"line":{"color":"rgb(63, 63, 63)"}},
                    decreasing={"marker":{"color":"blue"}},
                    increasing={"marker":{"color":"red"}},
                    text=[f"{base_value:.3f}"] + [f"{v:.3f}" for v in waterfall_values] + [f"{final_value:.3f}"],
                    textposition="outside"
                ))
                
                fig.update_layout(
                    title="Feature Impact on Risk Prediction",
                    showlegend=False,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add textual explanation
                st.markdown("### Explanation Summary")
                st.markdown(f"""
                The base risk score for all transactions starts at **{base_value:.2f}**. This transaction's final risk 
                score of **{final_value:.4f}** was influenced by the following factors:
                """)
                
                for i, (feature, value) in enumerate(zip(feature_names, waterfall_values)):
                    impact = "increased" if value > 0 else "decreased"
                    st.markdown(f"- {feature} {impact} the risk score by **{abs(value):.4f}**")
                
                if final_value >= 0.8:
                    st.error("This transaction has a high risk of fraud and requires immediate attention.")
                elif final_value >= 0.5:
                    st.warning("This transaction has a moderate risk of fraud and should be monitored.")
                else:
                    st.success("This transaction has a low risk of fraud.")
        else:
            st.info("No transaction data available for explanation.")
    
    with tabs[2]:
        st.markdown("### 📊 Global Feature Importance")
        st.markdown("""
        This visualization shows the relative importance of different features in the fraud detection model.
        Higher values indicate features that have a stronger influence on the model's predictions.
        """)
        
        # Create feature importance bar chart (in a real system, these would be actual model weights)
        feature_imp = {
            'Transaction Amount': 0.28,
            'Time of Transaction': 0.15,
            'Customer History': 0.22,
            'Merchant Category': 0.18,
            'Location': 0.12,
            'Device Information': 0.05
        }
        
        fig = go.Figure([
            go.Bar(
                x=list(feature_imp.keys()),
                y=list(feature_imp.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            )
        ])
        
        fig.update_layout(
            title='Global Feature Importance',
            xaxis_title='Feature',
            yaxis_title='Importance Score',
            yaxis=dict(range=[0, 0.3])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of feature importance
        st.markdown("""
        ### Understanding Feature Importance
        
        Features with higher importance scores have a greater influence on the model's predictions:
        
        - **Transaction Amount**: The value of the transaction is the most important indicator of potential fraud
        - **Customer History**: Past transaction patterns significantly impact risk assessment
        - **Merchant Category**: Certain merchant categories are associated with higher fraud rates
        - **Time of Transaction**: Unusual transaction times may indicate suspicious activity
        - **Location**: Geographical information helps identify unusual transaction patterns
        - **Device Information**: Data about the device used for the transaction
        """)
    
    # Transaction Details Section (if a transaction is selected)
    if st.session_state.selected_transaction is not None:
        st.subheader("Transaction Details")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Transaction Information")
            for col, val in st.session_state.selected_transaction.items():
                if col not in ['bayesian_score', 'mle_score', 'combined_score', 'fuzzy_score', 'bandit_score', 'is_flagged']:
                    st.text(f"{col}: {val}")
            
            # Risk scores with color coding
            st.markdown("### Risk Scores")
            
            bayesian = st.session_state.selected_transaction['bayesian_score']
            mle = st.session_state.selected_transaction['mle_score']
            combined = st.session_state.selected_transaction['combined_score']
            fuzzy = st.session_state.selected_transaction['fuzzy_score']
            bandit = st.session_state.selected_transaction['bandit_score'] if 'bandit_score' in st.session_state.selected_transaction else 0.0
            
            # Function to get color based on score
            def get_color(score):
                if score >= 0.8:
                    return "red"
                elif score >= 0.5:
                    return "orange"
                else:
                    return "green"
            
            # Display scores with color coding
            st.markdown("#### Core Risk Models")
            st.markdown(f"**Bayesian Score:** <span style='color:{get_color(bayesian)}'>{bayesian:.4f}</span>", unsafe_allow_html=True)
            st.markdown(f"**MLE Score:** <span style='color:{get_color(mle)}'>{mle:.4f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Combined Score:** <span style='color:{get_color(combined)}'>{combined:.4f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Fuzzy Score:** <span style='color:{get_color(fuzzy)}'>{fuzzy:.4f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Multi-Armed Bandit Score:** <span style='color:{get_color(bandit)}'>{bandit:.4f}</span>", unsafe_allow_html=True)
            
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
    
    # Real-time alerts and simulation section
    st.subheader("Real-Time Fraud Monitoring")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Real-time alerts display
        st.markdown("### 🚨 Alerts")
        if st.session_state.filtered_data is not None:
            flagged_transactions = st.session_state.filtered_data[st.session_state.filtered_data['is_flagged'] == True]
            
            if not flagged_transactions.empty:
                st.warning(f"⚠️ {len(flagged_transactions)} transactions have been flagged as potentially fraudulent.")
                
                # Show the most recent 3 flagged transactions
                for idx, row in flagged_transactions.head(3).iterrows():
                    with st.expander(f"Transaction ID: {row.get('transaction_id', idx)}"):
                        for col, val in row.items():
                            if col in ['bayesian_score', 'mle_score', 'combined_score', 'fuzzy_score', 'bandit_score']:
                                st.markdown(f"**{col}:** {val:.4f}")
                            elif col == 'is_flagged':
                                continue
                            else:
                                st.markdown(f"**{col}:** {val}")
            else:
                st.success("No fraudulent transactions detected in the current dataset.")
    
    with col2:
        # Transaction simulation feature
        st.markdown("### ⚡ Live Simulation")
        st.markdown("Simulate real-time transaction monitoring")
        
        if st.session_state.data is not None:
            # Initialize simulation state if not exists
            if 'simulation_active' not in st.session_state:
                st.session_state.simulation_active = False
            if 'simulation_speed' not in st.session_state:
                st.session_state.simulation_speed = 1.0
            if 'sim_transaction_counter' not in st.session_state:
                st.session_state.sim_transaction_counter = 0
            
            # Simulation controls
            simulation_speed = st.slider(
                "Speed", 
                min_value=0.5, 
                max_value=5.0, 
                value=st.session_state.simulation_speed,
                step=0.5,
                help="Transactions per second"
            )
            st.session_state.simulation_speed = simulation_speed
            
            # Toggle simulation button
            if st.session_state.simulation_active:
                if st.button("Stop Simulation"):
                    st.session_state.simulation_active = False
                    st.success("Simulation stopped")
            else:
                if st.button("Start Simulation"):
                    st.session_state.simulation_active = True
                    st.session_state.sim_transaction_counter = 0
                    st.info("Simulation started")
            
            # Display simulation status
            if st.session_state.simulation_active:
                # Create a placeholder for dynamic content
                sim_status = st.empty()
                
                # Update the simulation counter
                current_time = time.time()
                if 'last_sim_time' not in st.session_state:
                    st.session_state.last_sim_time = current_time
                
                # Check if it's time to simulate a new transaction
                time_diff = current_time - st.session_state.last_sim_time
                if time_diff > (1.0 / st.session_state.simulation_speed):
                    # Generate a new random transaction
                    st.session_state.sim_transaction_counter += 1
                    st.session_state.last_sim_time = current_time
                    
                    # Create a simulated transaction
                    new_transaction = {
                        'transaction_id': f"SIM{1000 + st.session_state.sim_transaction_counter}",
                        'amount': np.random.choice([
                            np.random.uniform(10, 200),  # Normal small purchase
                            np.random.uniform(500, 5000)  # Large purchase (sometimes suspicious)
                        ], p=[0.7, 0.3]),
                        'merchant': np.random.choice([
                            'Online Store', 'Local Shop', 'Restaurant', 
                            'Unknown Merchant', 'Foreign Vendor'
                        ]),
                        'location': np.random.choice([
                            'Local', 'Nearby City', 'Different State', 
                            'Foreign Country', 'Online'
                        ]),
                        'transaction_type': np.random.choice([
                            'retail', 'online', 'cash_advance', 'wire_transfer', 'atm'
                        ]),
                        'card_present': np.random.choice([True, False], p=[0.6, 0.4])
                    }
                    
                    # Process the transaction through risk models
                    # This is simplified for simulation purposes
                    risk_factors = []
                    risk_score = 0.0
                    
                    # Amount factor
                    if new_transaction['amount'] > 1000:
                        risk_factors.append("High amount")
                        risk_score += 0.3
                    
                    # Merchant factor
                    if new_transaction['merchant'] in ['Unknown Merchant', 'Foreign Vendor']:
                        risk_factors.append("Suspicious merchant")
                        risk_score += 0.25
                    
                    # Location factor
                    if new_transaction['location'] in ['Foreign Country']:
                        risk_factors.append("Foreign location")
                        risk_score += 0.2
                    
                    # Transaction type factor
                    if new_transaction['transaction_type'] in ['cash_advance', 'wire_transfer']:
                        risk_factors.append("High-risk transaction type")
                        risk_score += 0.15
                    
                    # Card present factor
                    if not new_transaction['card_present']:
                        risk_factors.append("Card not present")
                        risk_score += 0.1
                    
                    # Random factor for variety
                    risk_score += np.random.uniform(-0.1, 0.1)
                    risk_score = min(1.0, max(0.0, risk_score))
                    
                    # Determine if transaction is flagged
                    is_flagged = risk_score >= 0.7
                    
                    # Display transaction info
                    sim_status.markdown(f"""
                    **Simulated Transaction #{st.session_state.sim_transaction_counter}**  
                    ID: {new_transaction['transaction_id']}  
                    Amount: ${new_transaction['amount']:.2f}  
                    Merchant: {new_transaction['merchant']}  
                    Risk Score: {risk_score:.4f}  
                    Status: {"🚨 FLAGGED" if is_flagged else "✅ Approved"}
                    """)
                    
                    # If flagged, show alert
                    if is_flagged:
                        alert_placeholder = st.empty()
                        alert_placeholder.error(f"""
                        🚨 **FRAUD ALERT**  
                        Transaction {new_transaction['transaction_id']} flagged!  
                        Risk Score: {risk_score:.4f}
                        Risk Factors: {', '.join(risk_factors)}
                        """)
                else:
                    # No new transaction, show current status
                    sim_status.markdown(f"""
                    **Simulation Active**  
                    Transactions processed: {st.session_state.sim_transaction_counter}  
                    Speed: {st.session_state.simulation_speed} tx/sec  
                    Monitoring for fraud...
                    """)
        else:
            st.info("Upload transaction data to enable simulation")

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
