"""Data loading and preprocessing for Bond Yield Curve Analysis System."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from fredapi import Fred
except ImportError:
    Fred = None

from ..utils.config import DataConfig
from ..utils.utils import setup_logging, validate_data_quality


class TreasuryDataLoader:
    """Loader for Treasury yield curve data from various sources."""
    
    def __init__(self, config: DataConfig, api_key: Optional[str] = None):
        """Initialize the data loader.
        
        Args:
            config: Data configuration
            api_key: FRED API key (optional)
        """
        self.config = config
        self.logger = setup_logging()
        self.fred = Fred(api_key=api_key) if api_key and Fred else None
        
        # Treasury symbol mappings
        self.treasury_symbols = {
            "DGS1MO": 1/12,   # 1 month
            "DGS3MO": 3/12,   # 3 months
            "DGS6MO": 6/12,   # 6 months
            "DGS1": 1,        # 1 year
            "DGS2": 2,        # 2 years
            "DGS3": 3,        # 3 years
            "DGS5": 5,        # 5 years
            "DGS7": 7,        # 7 years
            "DGS10": 10,      # 10 years
            "DGS20": 20,      # 20 years
            "DGS30": 30,      # 30 years
        }
    
    def load_treasury_data(self) -> pd.DataFrame:
        """Load Treasury yield curve data.
        
        Returns:
            DataFrame with Treasury yield data
        """
        if self.config.source == "fred":
            return self._load_from_fred()
        elif self.config.source == "synthetic":
            return self._generate_synthetic_data()
        elif self.config.source == "csv":
            return self._load_from_csv()
        else:
            raise ValueError(f"Unknown data source: {self.config.source}")
    
    def _load_from_fred(self) -> pd.DataFrame:
        """Load data from FRED API."""
        if self.fred is None:
            self.logger.warning("FRED API key not provided, falling back to synthetic data")
            return self._generate_synthetic_data()
        
        self.logger.info("Loading Treasury data from FRED API")
        
        data_frames = []
        for symbol in self.config.symbols:
            try:
                data = self.fred.get_series(
                    symbol,
                    start=self.config.start_date,
                    end=self.config.end_date
                )
                
                if not data.empty:
                    df = pd.DataFrame({
                        'date': data.index,
                        'symbol': symbol,
                        'yield': data.values,
                        'maturity': self.treasury_symbols.get(symbol, np.nan)
                    })
                    data_frames.append(df)
                    self.logger.info(f"Loaded {len(data)} observations for {symbol}")
                else:
                    self.logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error loading {symbol}: {e}")
        
        if not data_frames:
            self.logger.warning("No data loaded from FRED, falling back to synthetic data")
            return self._generate_synthetic_data()
        
        # Combine all data
        combined_data = pd.concat(data_frames, ignore_index=True)
        combined_data = combined_data.sort_values(['date', 'maturity'])
        
        # Validate data quality
        validation = validate_data_quality(
            combined_data,
            required_columns=['date', 'symbol', 'yield', 'maturity']
        )
        
        if not validation["is_valid"]:
            self.logger.warning(f"Data quality issues detected: {validation['issues']}")
        
        return combined_data
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic Treasury yield curve data."""
        self.logger.info("Generating synthetic Treasury yield data")
        
        # Create date range
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter to business days only
        dates = dates[dates.weekday < 5]
        
        data_frames = []
        
        for symbol in self.config.symbols:
            maturity = self.treasury_symbols.get(symbol, 1.0)
            
            # Generate realistic yield curve with some randomness
            base_yield = self._get_base_yield(maturity)
            
            # Add time series dynamics
            np.random.seed(42)  # For reproducibility
            n_obs = len(dates)
            
            # Generate AR(1) process for yield changes
            yield_changes = np.random.normal(0, 0.001, n_obs)
            for i in range(1, n_obs):
                yield_changes[i] = 0.95 * yield_changes[i-1] + yield_changes[i]
            
            # Calculate yields
            yields = base_yield + np.cumsum(yield_changes)
            
            # Ensure yields are positive
            yields = np.maximum(yields, 0.01)
            
            df = pd.DataFrame({
                'date': dates,
                'symbol': symbol,
                'yield': yields,
                'maturity': maturity
            })
            data_frames.append(df)
        
        combined_data = pd.concat(data_frames, ignore_index=True)
        combined_data = combined_data.sort_values(['date', 'maturity'])
        
        return combined_data
    
    def _get_base_yield(self, maturity: float) -> float:
        """Get base yield for a given maturity (realistic yield curve shape)."""
        # Typical upward-sloping yield curve
        if maturity <= 0.25:  # 3 months or less
            return 0.5 + maturity * 2
        elif maturity <= 2:   # 2 years or less
            return 1.0 + maturity * 0.5
        elif maturity <= 10:  # 10 years or less
            return 2.0 + (maturity - 2) * 0.3
        else:  # Longer maturities
            return 4.0 + (maturity - 10) * 0.1
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load data from CSV file."""
        # This would be implemented to load from a local CSV file
        # For now, fall back to synthetic data
        self.logger.info("CSV loading not implemented, using synthetic data")
        return self._generate_synthetic_data()
    
    def create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create forward-looking labels for the data.
        
        Args:
            data: Input Treasury data
            
        Returns:
            DataFrame with labels
        """
        self.logger.info("Creating forward-looking labels")
        
        labels_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            
            # Calculate forward returns
            symbol_data['forward_return_1m'] = symbol_data['yield'].shift(-22) - symbol_data['yield']  # ~1 month
            symbol_data['forward_return_3m'] = symbol_data['yield'].shift(-66) - symbol_data['yield']  # ~3 months
            symbol_data['forward_return_6m'] = symbol_data['yield'].shift(-132) - symbol_data['yield']  # ~6 months
            
            # Calculate yield changes
            symbol_data['yield_change_1m'] = symbol_data['yield'].diff(22)
            symbol_data['yield_change_3m'] = symbol_data['yield'].diff(66)
            
            labels_data.append(symbol_data)
        
        # Combine all labels
        labels_df = pd.concat(labels_data, ignore_index=True)
        
        # Calculate curve metrics
        curve_metrics = self._calculate_curve_metrics(labels_df)
        labels_df = labels_df.merge(curve_metrics, on='date', how='left')
        
        # Add inversion indicator
        labels_df['curve_inversion'] = labels_df['curve_slope'] < -0.5
        
        return labels_df
    
    def _calculate_curve_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate yield curve metrics for each date."""
        curve_data = []
        
        for date in data['date'].unique():
            date_data = data[data['date'] == date]
            
            if len(date_data) >= 2:
                # Calculate slope (10Y - 2Y)
                yields_2y = date_data[date_data['symbol'] == 'DGS2']['yield']
                yields_10y = date_data[date_data['symbol'] == 'DGS10']['yield']
                
                if not yields_2y.empty and not yields_10y.empty:
                    slope = yields_10y.iloc[0] - yields_2y.iloc[0]
                    
                    curve_data.append({
                        'date': date,
                        'curve_slope': slope,
                        'curve_level': date_data['yield'].mean(),
                        'curve_volatility': date_data['yield'].std()
                    })
        
        return pd.DataFrame(curve_data)
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for modeling.
        
        Args:
            data: Raw data
            
        Returns:
            Preprocessed data
        """
        self.logger.info("Preprocessing data")
        
        # Remove missing values
        data = data.dropna()
        
        # Convert yields to decimal form
        data['yield'] = data['yield'] / 100
        
        # Add time features
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.dayofweek
        data['day_of_year'] = data['date'].dt.dayofyear
        
        # Add lagged features
        for symbol in data['symbol'].unique():
            symbol_mask = data['symbol'] == symbol
            symbol_data = data[symbol_mask].sort_values('date')
            
            # Add lagged yields
            for lag in [1, 5, 22]:  # 1 day, 1 week, 1 month
                data.loc[symbol_mask, f'yield_lag_{lag}'] = symbol_data['yield'].shift(lag)
            
            # Add rolling statistics
            for window in [5, 22, 66]:  # 1 week, 1 month, 3 months
                data.loc[symbol_mask, f'yield_ma_{window}'] = symbol_data['yield'].rolling(window).mean()
                data.loc[symbol_mask, f'yield_std_{window}'] = symbol_data['yield'].rolling(window).std()
        
        # Remove rows with NaN values from lagged features
        data = data.dropna()
        
        return data
    
    def create_train_test_split(
        self, 
        data: pd.DataFrame, 
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-based train/validation/test splits.
        
        Args:
            data: Input data
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            
        Returns:
            Tuple of (train_data, validation_data, test_data)
        """
        self.logger.info("Creating time-based train/validation/test splits")
        
        # Sort by date
        data = data.sort_values('date')
        
        # Calculate split points
        n_total = len(data)
        n_test = int(n_total * test_size)
        n_validation = int(n_total * validation_size)
        n_train = n_total - n_test - n_validation
        
        # Split data
        train_data = data.iloc[:n_train].copy()
        validation_data = data.iloc[n_train:n_train + n_validation].copy()
        test_data = data.iloc[n_train + n_validation:].copy()
        
        self.logger.info(f"Train: {len(train_data)} samples")
        self.logger.info(f"Validation: {len(validation_data)} samples")
        self.logger.info(f"Test: {len(test_data)} samples")
        
        return train_data, validation_data, test_data
