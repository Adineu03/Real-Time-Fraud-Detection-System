import pandas as pd
import numpy as np
import io
import datetime

def process_uploaded_file(uploaded_file):
    """
    Process the uploaded transaction file and return a clean DataFrame
    
    Parameters:
    uploaded_file (UploadedFile): The file uploaded by the user
    
    Returns:
    pd.DataFrame: Processed DataFrame with transaction data
    """
    # Get file extension
    file_extension = uploaded_file.name.split('.')[-1]
    
    # Read file based on extension
    if file_extension.lower() == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension.lower() in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Basic data cleaning and preprocessing
    df = clean_transaction_data(df)
    
    return df

def clean_transaction_data(df):
    """
    Clean and preprocess transaction data
    
    Parameters:
    df (pd.DataFrame): Raw transaction data
    
    Returns:
    pd.DataFrame: Cleaned and preprocessed data
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date columns to datetime if they exist
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            # If conversion fails, leave as is
            pass
    
    # Handle missing values
    df = df.fillna({
        # Fill numeric columns with 0
        col: 0 for col in df.select_dtypes(include=[np.number]).columns
    })
    
    # For non-numeric columns, fill with 'Unknown'
    df = df.fillna({
        col: 'Unknown' for col in df.select_dtypes(exclude=[np.number]).columns
    })
    
    return df

def filter_dataframe(df, filter_conditions, show_flagged_only=False):
    """
    Filter the dataframe based on user-selected conditions
    
    Parameters:
    df (pd.DataFrame): Transaction data to filter
    filter_conditions (dict): Dictionary of filter conditions
    show_flagged_only (bool): Whether to show only flagged transactions
    
    Returns:
    pd.DataFrame: Filtered dataframe
    """
    # Make a copy to avoid modifying the original
    filtered_df = df.copy()
    
    # Apply date filter if exists
    if 'date' in filter_conditions and 'date' in filtered_df.columns:
        start_date, end_date = filter_conditions['date']
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    # Apply amount filter if exists
    if 'amount' in filter_conditions and 'amount' in filtered_df.columns:
        min_amount, max_amount = filter_conditions['amount']
        filtered_df = filtered_df[
            (filtered_df['amount'] >= min_amount) & 
            (filtered_df['amount'] <= max_amount)
        ]
    
    # Apply risk threshold filter
    if 'risk_threshold' in filter_conditions:
        threshold = filter_conditions['risk_threshold']
        filtered_df = filtered_df[filtered_df['combined_score'] >= threshold]
    
    # Apply flagged filter if requested
    if show_flagged_only:
        filtered_df = filtered_df[filtered_df['is_flagged'] == True]
    
    return filtered_df
