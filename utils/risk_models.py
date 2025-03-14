import numpy as np
import pandas as pd
from scipy import stats
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def calculate_bayesian_score(df):
    """
    Calculate Bayesian probability scores for fraud detection
    
    Parameters:
    df (pd.DataFrame): Transaction data
    
    Returns:
    pd.Series: Bayesian probability scores for each transaction
    """
    # In a real system, this would use actual Bayesian probability calculations
    # For this example, we'll create a score based on available transaction features
    
    # Initialize scores with random base (would use actual priors in a real system)
    scores = np.random.beta(2, 5, size=len(df))  # Beta distribution skewed toward lower values
    
    # Adjust scores based on transaction features if they exist
    if 'amount' in df.columns:
        # Higher amounts increase risk (normalize to 0-1 range)
        amount_factor = df['amount'] / df['amount'].max()
        scores = scores + (amount_factor * 0.3)  # Weighted contribution

    # If transaction time exists, transactions at odd hours are riskier
    if 'time' in df.columns or 'date' in df.columns:
        try:
            # Try to extract hour information
            if 'time' in df.columns:
                # Assuming time is in a string format like "HH:MM:SS"
                hours = df['time'].astype(str).str.split(':', expand=True)[0].astype(float)
            else:
                # Extract hour from datetime
                hours = df['date'].dt.hour
                
            # Transactions between 1 AM and 5 AM are higher risk
            night_factor = ((hours >= 1) & (hours <= 5)).astype(float) * 0.2
            scores = scores + night_factor
        except:
            # If time extraction fails, continue without this factor
            pass
    
    # Clip values to ensure they're between 0 and 1
    scores = np.clip(scores, 0, 1)
    
    return scores

def calculate_mle_score(df):
    """
    Calculate MLE-based probability scores for fraud detection
    
    Parameters:
    df (pd.DataFrame): Transaction data
    
    Returns:
    pd.Series: MLE-based probability scores for each transaction
    """
    # In a real system, this would use actual Maximum Likelihood Estimation
    # For this example, we'll create a different score based on transaction features
    
    # Initialize with slightly different distribution than Bayesian
    scores = np.random.beta(2, 7, size=len(df))  # More conservative base scores
    
    # Adjust based on features (with different weightings than Bayesian model)
    if 'amount' in df.columns:
        # Apply a non-linear transformation to amount
        amount_factor = np.log1p(df['amount']) / np.log1p(df['amount'].max())
        scores = scores + (amount_factor * 0.25)
    
    # Location-based risk if available
    if 'location' in df.columns:
        # High-risk locations increase score (in real system would use actual location risk data)
        # Here we're just using a random factor based on location strings
        location_hash = df['location'].astype(str).apply(hash)
        location_factor = (location_hash % 100) / 100 * 0.15
        scores = scores + location_factor
    
    # Merchant type risk if available
    if 'merchant' in df.columns or 'merchant_category' in df.columns:
        merchant_col = 'merchant' if 'merchant' in df.columns else 'merchant_category'
        # Similar random approach for example purposes
        merchant_hash = df[merchant_col].astype(str).apply(hash)
        merchant_factor = (merchant_hash % 100) / 100 * 0.2
        scores = scores + merchant_factor
    
    # Clip values to ensure they're between 0 and 1
    scores = np.clip(scores, 0, 1)
    
    return scores

def calculate_combined_score(bayesian_scores, mle_scores):
    """
    Calculate combined risk scores using both Bayesian and MLE scores
    
    Parameters:
    bayesian_scores (pd.Series): Bayesian probability scores
    mle_scores (pd.Series): MLE-based probability scores
    
    Returns:
    pd.Series: Combined risk scores
    """
    # Simple weighted average (would use more sophisticated methods in a real system)
    combined_scores = (bayesian_scores * 0.45) + (mle_scores * 0.55)
    
    # Ensure values are between 0 and 1
    combined_scores = np.clip(combined_scores, 0, 1)
    
    return combined_scores

def calculate_fuzzy_score(combined_scores, df):
    """
    Apply fuzzy logic to refine combined risk scores
    
    Parameters:
    combined_scores (pd.Series): Combined risk scores from Bayesian and MLE models
    df (pd.DataFrame): Transaction data
    
    Returns:
    pd.Series: Fuzzy logic adjusted risk scores
    """
    # In a real system, this would use actual fuzzy logic rules
    # For this example, we'll create an adjusted score with some randomness to simulate fuzzy logic
    
    # Start with the combined score
    fuzzy_scores = combined_scores.copy()
    
    # Add small adjustments to simulate fuzzy logic refinement
    # Slightly increase scores that are near the threshold (0.7-0.8 range)
    borderline_mask = (combined_scores >= 0.65) & (combined_scores <= 0.75)
    fuzzy_scores[borderline_mask] = fuzzy_scores[borderline_mask] + 0.1
    
    # Slightly decrease very high scores to reduce false positives
    high_mask = combined_scores > 0.9
    fuzzy_scores[high_mask] = fuzzy_scores[high_mask] - 0.05
    
    # Add a small random component to simulate the "fuzziness"
    random_adjustment = np.random.normal(0, 0.02, size=len(fuzzy_scores))
    fuzzy_scores = fuzzy_scores + random_adjustment
    
    # Clip values to ensure they're between 0 and 1
    fuzzy_scores = np.clip(fuzzy_scores, 0, 1)
    
    return fuzzy_scores

def flag_transactions(df, threshold=0.8):
    """
    Flag transactions as potentially fraudulent based on fuzzy risk score
    
    Parameters:
    df (pd.DataFrame): Transaction data with risk scores
    threshold (float): Threshold for flagging transactions
    
    Returns:
    pd.DataFrame: DataFrame with added is_flagged column
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Flag transactions based on fuzzy risk score
    df['is_flagged'] = df['fuzzy_score'] >= threshold
    
    return df
