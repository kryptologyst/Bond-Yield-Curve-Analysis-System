"""Tests for Bond Yield Curve Analysis System."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.loader import TreasuryDataLoader
from src.models.forecasting import ARIMAModel, VARModel, RegimeSwitchingModel
from src.risk.analyzer import YieldCurveRiskAnalyzer
from src.utils.config import DataConfig, ModelConfig, RiskConfig, SystemConfig
from src.utils.utils import (
    calculate_yield_curve_metrics,
    detect_yield_curve_inversion,
    calculate_risk_metrics,
    validate_data_quality
)


class TestDataLoader:
    """Test cases for TreasuryDataLoader."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        config = DataConfig(source="synthetic")
        loader = TreasuryDataLoader(config)
        
        data = loader.load_treasury_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'date' in data.columns
        assert 'symbol' in data.columns
        assert 'yield' in data.columns
        assert 'maturity' in data.columns
    
    def test_data_preprocessing(self):
        """Test data preprocessing."""
        config = DataConfig(source="synthetic")
        loader = TreasuryDataLoader(config)
        
        data = loader.load_treasury_data()
        processed_data = loader.preprocess_data(data)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        assert not processed_data.isnull().any().any()
    
    def test_train_test_split(self):
        """Test train-test split functionality."""
        config = DataConfig(source="synthetic")
        loader = TreasuryDataLoader(config)
        
        data = loader.load_treasury_data()
        train_data, val_data, test_data = loader.create_train_test_split(data)
        
        assert isinstance(train_data, pd.DataFrame)
        assert isinstance(val_data, pd.DataFrame)
        assert isinstance(test_data, pd.DataFrame)
        assert len(train_data) + len(val_data) + len(test_data) == len(data)


class TestForecastingModels:
    """Test cases for forecasting models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        symbols = ['DGS2', 'DGS5', 'DGS10', 'DGS30']
        
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'yield': 2.0 + np.random.normal(0, 0.1),
                    'maturity': {'DGS2': 2, 'DGS5': 5, 'DGS10': 10, 'DGS30': 30}[symbol]
                })
        
        return pd.DataFrame(data)
    
    def test_arima_model(self, sample_data):
        """Test ARIMA model."""
        config = ModelConfig(type="arima")
        model = ARIMAModel(config)
        
        # Fit model
        model.fit(sample_data)
        assert model.is_fitted
        
        # Make predictions
        predictions = model.predict(horizon=10)
        assert 'predictions' in predictions
        assert 'confidence_intervals' in predictions
    
    def test_var_model(self, sample_data):
        """Test VAR model."""
        config = ModelConfig(type="var")
        model = VARModel(config)
        
        # Fit model
        model.fit(sample_data)
        assert model.is_fitted
        
        # Make predictions
        predictions = model.predict(horizon=10)
        assert 'predictions' in predictions
        assert 'confidence_intervals' in predictions
    
    def test_regime_switching_model(self, sample_data):
        """Test regime switching model."""
        config = ModelConfig(type="regime_switching")
        model = RegimeSwitchingModel(config)
        
        # Fit model
        model.fit(sample_data)
        assert model.is_fitted
        
        # Make predictions
        predictions = model.predict(horizon=10)
        assert 'predictions' in predictions
        assert 'confidence_intervals' in predictions


class TestRiskAnalyzer:
    """Test cases for risk analyzer."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns for testing."""
        return np.random.normal(0.001, 0.02, 1000)
    
    @pytest.fixture
    def sample_yield_data(self):
        """Create sample yield curve data."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        symbols = ['DGS2', 'DGS10']
        
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'yield': 2.0 + np.random.normal(0, 0.1),
                    'maturity': {'DGS2': 2, 'DGS10': 10}[symbol]
                })
        
        return pd.DataFrame(data)
    
    def test_var_calculation(self, sample_returns):
        """Test VaR calculation."""
        config = RiskConfig()
        analyzer = YieldCurveRiskAnalyzer(config)
        
        var_results = analyzer.calculate_var(sample_returns)
        
        assert 'var_95' in var_results
        assert 'var_99' in var_results
        assert var_results['var_95'] > var_results['var_99']
    
    def test_expected_shortfall(self, sample_returns):
        """Test Expected Shortfall calculation."""
        config = RiskConfig()
        analyzer = YieldCurveRiskAnalyzer(config)
        
        es_results = analyzer.calculate_expected_shortfall(sample_returns)
        
        assert 'es_95' in es_results
        assert 'es_99' in es_results
    
    def test_maximum_drawdown(self, sample_returns):
        """Test maximum drawdown calculation."""
        config = RiskConfig()
        analyzer = YieldCurveRiskAnalyzer(config)
        
        drawdown_results = analyzer.calculate_maximum_drawdown(sample_returns)
        
        assert 'max_drawdown' in drawdown_results
        assert drawdown_results['max_drawdown'] <= 0
    
    def test_inversion_detection(self, sample_yield_data):
        """Test yield curve inversion detection."""
        config = RiskConfig()
        analyzer = YieldCurveRiskAnalyzer(config)
        
        inversion_results = analyzer.detect_yield_curve_inversion(sample_yield_data)
        
        assert 'is_inverted' in inversion_results
        assert 'slope' in inversion_results
        assert 'severity' in inversion_results


class TestUtils:
    """Test cases for utility functions."""
    
    def test_yield_curve_metrics(self):
        """Test yield curve metrics calculation."""
        yields = np.array([1.5, 2.0, 2.5, 3.0])
        maturities = np.array([1, 2, 5, 10])
        
        metrics = calculate_yield_curve_metrics(yields, maturities)
        
        assert 'level' in metrics
        assert 'slope' in metrics
        assert 'curvature' in metrics
        assert 'volatility' in metrics
        assert 'inverted' in metrics
    
    def test_inversion_detection(self):
        """Test inversion detection."""
        # Normal curve (upward sloping)
        yields_normal = np.array([1.0, 1.5, 2.0, 2.5])
        maturities = np.array([1, 2, 5, 10])
        
        is_inverted_normal = detect_yield_curve_inversion(yields_normal, maturities)
        assert not is_inverted_normal
        
        # Inverted curve (downward sloping)
        yields_inverted = np.array([3.0, 2.5, 2.0, 1.5])
        
        is_inverted_inverted = detect_yield_curve_inversion(yields_inverted, maturities)
        assert is_inverted_inverted
    
    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        returns = np.random.normal(0.001, 0.02, 1000)
        
        metrics = calculate_risk_metrics(returns)
        
        assert 'mean_return' in metrics
        assert 'volatility' in metrics
        assert 'var_95' in metrics
        assert 'var_99' in metrics
        assert 'max_drawdown' in metrics
    
    def test_data_validation(self):
        """Test data validation."""
        # Valid data
        valid_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100),
            'symbol': ['DGS2'] * 100,
            'yield': np.random.normal(2.0, 0.1, 100)
        })
        
        validation = validate_data_quality(valid_data, ['date', 'symbol', 'yield'])
        assert validation['is_valid']
        
        # Invalid data (missing columns)
        invalid_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100),
            'symbol': ['DGS2'] * 100
        })
        
        validation = validate_data_quality(invalid_data, ['date', 'symbol', 'yield'])
        assert not validation['is_valid']


class TestConfiguration:
    """Test cases for configuration management."""
    
    def test_data_config(self):
        """Test DataConfig."""
        config = DataConfig()
        
        assert config.source == "fred"
        assert isinstance(config.symbols, list)
        assert len(config.symbols) > 0
    
    def test_model_config(self):
        """Test ModelConfig."""
        config = ModelConfig()
        
        assert config.type == "var"
        assert config.horizon == 30
        assert isinstance(config.confidence_levels, list)
    
    def test_risk_config(self):
        """Test RiskConfig."""
        config = RiskConfig()
        
        assert config.var_horizon == 1
        assert isinstance(config.var_confidence_levels, list)
        assert isinstance(config.stress_scenarios, list)
    
    def test_system_config(self):
        """Test SystemConfig."""
        config = SystemConfig(
            data=DataConfig(),
            model=ModelConfig(),
            risk=RiskConfig(),
            backtest=None
        )
        
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.risk, RiskConfig)


if __name__ == "__main__":
    pytest.main([__file__])
