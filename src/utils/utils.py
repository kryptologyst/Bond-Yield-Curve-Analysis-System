"""Utility functions for Bond Yield Curve Analysis System."""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device_preference: Device preference ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device object
    """
    if device_preference == "cpu":
        return torch.device("cpu")
    elif device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def calculate_yield_curve_metrics(
    yields: Union[np.ndarray, pd.Series],
    maturities: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """Calculate key yield curve metrics.
    
    Args:
        yields: Yield rates for different maturities
        maturities: Corresponding maturities in years
        
    Returns:
        Dictionary containing yield curve metrics
    """
    yields = np.array(yields)
    maturities = np.array(maturities)
    
    # Sort by maturity
    sort_idx = np.argsort(maturities)
    yields = yields[sort_idx]
    maturities = maturities[sort_idx]
    
    metrics = {}
    
    # Level (average yield)
    metrics["level"] = np.mean(yields)
    
    # Slope (10Y - 2Y spread)
    if len(yields) >= 2:
        # Find closest to 2Y and 10Y
        idx_2y = np.argmin(np.abs(maturities - 2))
        idx_10y = np.argmin(np.abs(maturities - 10))
        metrics["slope"] = yields[idx_10y] - yields[idx_2y]
    else:
        metrics["slope"] = np.nan
    
    # Curvature (2Y + 10Y - 2*5Y)
    if len(yields) >= 3:
        idx_2y = np.argmin(np.abs(maturities - 2))
        idx_5y = np.argmin(np.abs(maturities - 5))
        idx_10y = np.argmin(np.abs(maturities - 10))
        metrics["curvature"] = yields[idx_2y] + yields[idx_10y] - 2 * yields[idx_5y]
    else:
        metrics["curvature"] = np.nan
    
    # Volatility
    metrics["volatility"] = np.std(yields)
    
    # Range (max - min)
    metrics["range"] = np.max(yields) - np.min(yields)
    
    # Inversion indicator
    metrics["inverted"] = metrics["slope"] < 0
    
    return metrics


def detect_yield_curve_inversion(
    yields: Union[np.ndarray, pd.Series],
    maturities: Union[np.ndarray, pd.Series],
    threshold: float = -0.5
) -> bool:
    """Detect yield curve inversion.
    
    Args:
        yields: Yield rates for different maturities
        maturities: Corresponding maturities in years
        threshold: Inversion threshold in basis points
        
    Returns:
        True if yield curve is inverted
    """
    metrics = calculate_yield_curve_metrics(yields, maturities)
    return metrics["slope"] < threshold


def calculate_forward_rates(
    spot_rates: Union[np.ndarray, pd.Series],
    maturities: Union[np.ndarray, pd.Series]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate forward rates from spot rates.
    
    Args:
        spot_rates: Spot rates for different maturities
        maturities: Corresponding maturities in years
        
    Returns:
        Tuple of (forward_maturities, forward_rates)
    """
    spot_rates = np.array(spot_rates)
    maturities = np.array(maturities)
    
    # Sort by maturity
    sort_idx = np.argsort(maturities)
    spot_rates = spot_rates[sort_idx]
    maturities = maturities[sort_idx]
    
    forward_maturities = []
    forward_rates = []
    
    for i in range(1, len(maturities)):
        t1 = maturities[i-1]
        t2 = maturities[i]
        r1 = spot_rates[i-1]
        r2 = spot_rates[i]
        
        # Forward rate from t1 to t2
        forward_rate = (r2 * t2 - r1 * t1) / (t2 - t1)
        
        forward_maturities.append((t1 + t2) / 2)  # Midpoint
        forward_rates.append(forward_rate)
    
    return np.array(forward_maturities), np.array(forward_rates)


def calculate_duration(
    yields: Union[np.ndarray, pd.Series],
    maturities: Union[np.ndarray, pd.Series],
    coupon_rate: float = 0.0
) -> float:
    """Calculate Macaulay duration for a bond portfolio.
    
    Args:
        yields: Yield rates for different maturities
        maturities: Corresponding maturities in years
        coupon_rate: Coupon rate (assumed constant across maturities)
        
    Returns:
        Macaulay duration
    """
    yields = np.array(yields)
    maturities = np.array(maturities)
    
    # Calculate present values
    pv_coupons = coupon_rate * np.exp(-yields * maturities)
    pv_principal = np.exp(-yields * maturities)
    total_pv = pv_coupons + pv_principal
    
    # Calculate weighted average maturity
    duration = np.sum(maturities * total_pv) / np.sum(total_pv)
    
    return duration


def calculate_convexity(
    yields: Union[np.ndarray, pd.Series],
    maturities: Union[np.ndarray, pd.Series],
    coupon_rate: float = 0.0
) -> float:
    """Calculate convexity for a bond portfolio.
    
    Args:
        yields: Yield rates for different maturities
        maturities: Corresponding maturities in years
        coupon_rate: Coupon rate (assumed constant across maturities)
        
    Returns:
        Convexity measure
    """
    yields = np.array(yields)
    maturities = np.array(maturities)
    
    # Calculate present values
    pv_coupons = coupon_rate * np.exp(-yields * maturities)
    pv_principal = np.exp(-yields * maturities)
    total_pv = pv_coupons + pv_principal
    
    # Calculate convexity
    convexity = np.sum(maturities**2 * total_pv) / np.sum(total_pv)
    
    return convexity


def bootstrap_yield_curve(
    market_prices: Union[np.ndarray, pd.Series],
    maturities: Union[np.ndarray, pd.Series],
    face_value: float = 100.0
) -> np.ndarray:
    """Bootstrap zero-coupon yield curve from market prices.
    
    Args:
        market_prices: Market prices of bonds
        maturities: Corresponding maturities in years
        face_value: Face value of bonds
        
    Returns:
        Bootstrapped zero-coupon yields
    """
    market_prices = np.array(market_prices)
    maturities = np.array(maturities)
    
    # Sort by maturity
    sort_idx = np.argsort(maturities)
    market_prices = market_prices[sort_idx]
    maturities = maturities[sort_idx]
    
    yields = np.zeros_like(maturities)
    
    for i, (price, maturity) in enumerate(zip(market_prices, maturities)):
        # Calculate yield to maturity
        yields[i] = -np.log(price / face_value) / maturity
    
    return yields


def calculate_risk_metrics(
    returns: Union[np.ndarray, pd.Series],
    confidence_levels: List[float] = None
) -> Dict[str, float]:
    """Calculate risk metrics for a return series.
    
    Args:
        returns: Return series
        confidence_levels: Confidence levels for VaR calculation
        
    Returns:
        Dictionary containing risk metrics
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]
    
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]  # Remove NaN values
    
    metrics = {}
    
    # Basic statistics
    metrics["mean_return"] = np.mean(returns)
    metrics["volatility"] = np.std(returns)
    metrics["skewness"] = stats.skew(returns)
    metrics["kurtosis"] = stats.kurtosis(returns)
    
    # Risk metrics
    metrics["var_95"] = np.percentile(returns, 5)
    metrics["var_99"] = np.percentile(returns, 1)
    
    # Expected Shortfall (Conditional VaR)
    metrics["es_95"] = np.mean(returns[returns <= metrics["var_95"]])
    metrics["es_99"] = np.mean(returns[returns <= metrics["var_99"]])
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    metrics["max_drawdown"] = np.min(drawdowns)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    metrics["sharpe_ratio"] = metrics["mean_return"] / metrics["volatility"] if metrics["volatility"] > 0 else 0
    
    return metrics


def validate_data_quality(
    data: pd.DataFrame,
    required_columns: List[str],
    date_column: str = "date"
) -> Dict[str, Any]:
    """Validate data quality and detect potential issues.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        date_column: Name of the date column
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        "is_valid": True,
        "issues": [],
        "statistics": {}
    }
    
    # Check required columns
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        validation_results["is_valid"] = False
        validation_results["issues"].append(f"Missing columns: {missing_columns}")
    
    # Check for duplicate dates
    if date_column in data.columns:
        duplicate_dates = data[date_column].duplicated().sum()
        if duplicate_dates > 0:
            validation_results["issues"].append(f"Found {duplicate_dates} duplicate dates")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        validation_results["issues"].append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
    
    # Check for infinite values
    infinite_values = np.isinf(data.select_dtypes(include=[np.number])).sum()
    if infinite_values.sum() > 0:
        validation_results["issues"].append(f"Infinite values found: {infinite_values[infinite_values > 0].to_dict()}")
    
    # Basic statistics
    validation_results["statistics"] = {
        "shape": data.shape,
        "date_range": (data[date_column].min(), data[date_column].max()) if date_column in data.columns else None,
        "missing_values": missing_values.to_dict(),
        "data_types": data.dtypes.to_dict()
    }
    
    return validation_results
