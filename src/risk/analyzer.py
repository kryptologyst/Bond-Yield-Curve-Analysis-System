"""Risk analysis tools for Bond Yield Curve Analysis System."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..utils.config import RiskConfig
from ..utils.utils import setup_logging, calculate_risk_metrics


class YieldCurveRiskAnalyzer:
    """Risk analysis tools for yield curve data."""
    
    def __init__(self, config: RiskConfig):
        """Initialize the risk analyzer.
        
        Args:
            config: Risk configuration
        """
        self.config = config
        self.logger = setup_logging()
    
    def calculate_var(
        self, 
        returns: Union[np.ndarray, pd.Series], 
        confidence_levels: List[float] = None
    ) -> Dict[str, float]:
        """Calculate Value at Risk (VaR) for return series.
        
        Args:
            returns: Return series
            confidence_levels: Confidence levels for VaR calculation
            
        Returns:
            Dictionary containing VaR values
        """
        if confidence_levels is None:
            confidence_levels = self.config.var_confidence_levels
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        var_results = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            var_value = np.percentile(returns, alpha * 100)
            var_results[f'var_{int(conf_level*100)}'] = var_value
        
        return var_results
    
    def calculate_expected_shortfall(
        self, 
        returns: Union[np.ndarray, pd.Series], 
        confidence_levels: List[float] = None
    ) -> Dict[str, float]:
        """Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Return series
            confidence_levels: Confidence levels for ES calculation
            
        Returns:
            Dictionary containing ES values
        """
        if confidence_levels is None:
            confidence_levels = self.config.var_confidence_levels
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        es_results = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            var_value = np.percentile(returns, alpha * 100)
            tail_returns = returns[returns <= var_value]
            es_value = np.mean(tail_returns) if len(tail_returns) > 0 else var_value
            es_results[f'es_{int(conf_level*100)}'] = es_value
        
        return es_results
    
    def calculate_maximum_drawdown(self, returns: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate maximum drawdown.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary containing drawdown metrics
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = np.min(drawdowns)
        max_drawdown_idx = np.argmin(drawdowns)
        
        # Find recovery point
        recovery_idx = None
        if max_drawdown_idx < len(drawdowns) - 1:
            recovery_candidates = np.where(drawdowns[max_drawdown_idx:] >= 0)[0]
            if len(recovery_candidates) > 0:
                recovery_idx = max_drawdown_idx + recovery_candidates[0]
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_idx': max_drawdown_idx,
            'recovery_idx': recovery_idx,
            'drawdown_duration': recovery_idx - max_drawdown_idx if recovery_idx else None
        }
    
    def stress_test_yield_curve(
        self, 
        data: pd.DataFrame, 
        scenarios: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Perform stress testing on yield curve data.
        
        Args:
            data: Yield curve data
            scenarios: List of stress scenarios
            
        Returns:
            Dictionary containing stress test results
        """
        if scenarios is None:
            scenarios = self.config.stress_scenarios
        
        stress_results = {}
        
        for scenario in scenarios:
            self.logger.info(f"Running stress test: {scenario}")
            
            if scenario == "recession":
                stress_results[scenario] = self._recession_stress_test(data)
            elif scenario == "inflation_shock":
                stress_results[scenario] = self._inflation_shock_stress_test(data)
            elif scenario == "flight_to_quality":
                stress_results[scenario] = self._flight_to_quality_stress_test(data)
            else:
                self.logger.warning(f"Unknown stress scenario: {scenario}")
        
        return stress_results
    
    def _recession_stress_test(self, data: pd.DataFrame) -> Dict[str, float]:
        """Recession stress test: flattening yield curve."""
        # Calculate current curve slope
        current_slope = self._calculate_curve_slope(data)
        
        # Stress: flatten curve by 100 bps
        stress_slope = current_slope - 1.0
        
        return {
            'scenario': 'recession',
            'current_slope': current_slope,
            'stressed_slope': stress_slope,
            'impact': stress_slope - current_slope
        }
    
    def _inflation_shock_stress_test(self, data: pd.DataFrame) -> Dict[str, float]:
        """Inflation shock stress test: parallel shift up."""
        # Calculate current curve level
        current_level = data['yield'].mean()
        
        # Stress: parallel shift up by 200 bps
        stress_level = current_level + 2.0
        
        return {
            'scenario': 'inflation_shock',
            'current_level': current_level,
            'stressed_level': stress_level,
            'impact': stress_level - current_level
        }
    
    def _flight_to_quality_stress_test(self, data: pd.DataFrame) -> Dict[str, float]:
        """Flight to quality stress test: steepening curve."""
        # Calculate current curve slope
        current_slope = self._calculate_curve_slope(data)
        
        # Stress: steepen curve by 150 bps
        stress_slope = current_slope + 1.5
        
        return {
            'scenario': 'flight_to_quality',
            'current_slope': current_slope,
            'stressed_slope': stress_slope,
            'impact': stress_slope - current_slope
        }
    
    def _calculate_curve_slope(self, data: pd.DataFrame) -> float:
        """Calculate yield curve slope (10Y - 2Y)."""
        # Get 2Y and 10Y yields
        yield_2y = data[data['symbol'] == 'DGS2']['yield']
        yield_10y = data[data['symbol'] == 'DGS10']['yield']
        
        if yield_2y.empty or yield_10y.empty:
            return 0.0
        
        return yield_10y.iloc[0] - yield_2y.iloc[0]
    
    def detect_yield_curve_inversion(
        self, 
        data: pd.DataFrame, 
        threshold: float = None
    ) -> Dict[str, Any]:
        """Detect yield curve inversion.
        
        Args:
            data: Yield curve data
            threshold: Inversion threshold in basis points
            
        Returns:
            Dictionary containing inversion analysis
        """
        if threshold is None:
            threshold = self.config.inversion_threshold
        
        # Calculate curve slope
        slope = self._calculate_curve_slope(data)
        
        # Check for inversion
        is_inverted = slope < threshold
        
        # Calculate inversion severity
        severity = abs(slope) if is_inverted else 0
        
        return {
            'is_inverted': is_inverted,
            'slope': slope,
            'threshold': threshold,
            'severity': severity,
            'severity_level': self._classify_inversion_severity(severity)
        }
    
    def _classify_inversion_severity(self, severity: float) -> str:
        """Classify inversion severity."""
        if severity < 0.25:
            return "mild"
        elif severity < 0.5:
            return "moderate"
        elif severity < 1.0:
            return "severe"
        else:
            return "extreme"
    
    def calculate_portfolio_risk(
        self, 
        weights: Union[np.ndarray, pd.Series], 
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics.
        
        Args:
            weights: Portfolio weights
            returns: Asset returns
            cov_matrix: Covariance matrix (optional)
            
        Returns:
            Dictionary containing portfolio risk metrics
        """
        weights = np.array(weights)
        returns = np.array(returns)
        
        # Calculate portfolio return
        portfolio_return = np.mean(returns @ weights)
        
        # Calculate portfolio volatility
        if cov_matrix is not None:
            portfolio_variance = weights.T @ cov_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)
        else:
            portfolio_returns = returns @ weights
            portfolio_volatility = np.std(portfolio_returns)
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Calculate VaR
        portfolio_returns = returns @ weights
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        return {
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'var_99': var_99
        }
    
    def monte_carlo_simulation(
        self, 
        data: pd.DataFrame, 
        n_simulations: int = 10000,
        horizon: int = 30
    ) -> Dict[str, Any]:
        """Perform Monte Carlo simulation for yield curve forecasting.
        
        Args:
            data: Historical yield curve data
            n_simulations: Number of simulations
            horizon: Forecast horizon
            
        Returns:
            Dictionary containing simulation results
        """
        self.logger.info(f"Running Monte Carlo simulation with {n_simulations} paths")
        
        # Calculate historical statistics
        returns = data.groupby('symbol')['yield'].pct_change().dropna()
        
        simulation_results = {}
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            symbol_returns = symbol_data['yield'].pct_change().dropna()
            
            if len(symbol_returns) < 10:
                continue
            
            # Calculate historical statistics
            mean_return = symbol_returns.mean()
            std_return = symbol_returns.std()
            
            # Generate random paths
            random_paths = np.random.normal(
                mean_return, 
                std_return, 
                (n_simulations, horizon)
            )
            
            # Calculate final yields
            current_yield = symbol_data['yield'].iloc[-1]
            final_yields = current_yield * np.prod(1 + random_paths, axis=1)
            
            simulation_results[symbol] = {
                'mean_final_yield': np.mean(final_yields),
                'std_final_yield': np.std(final_yields),
                'var_95': np.percentile(final_yields, 5),
                'var_99': np.percentile(final_yields, 1),
                'paths': random_paths,
                'final_yields': final_yields
            }
        
        return simulation_results
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive risk analysis of yield curve data.
        
        Args:
            data: Yield curve data
            
        Returns:
            Dictionary containing comprehensive risk analysis
        """
        self.logger.info("Performing comprehensive risk analysis")
        
        analysis_results = {}
        
        # Calculate basic risk metrics
        returns = data.groupby('symbol')['yield'].pct_change().dropna()
        risk_metrics = calculate_risk_metrics(returns)
        analysis_results['risk_metrics'] = risk_metrics
        
        # Calculate VaR and ES
        var_results = self.calculate_var(returns)
        es_results = self.calculate_expected_shortfall(returns)
        analysis_results['var'] = var_results
        analysis_results['expected_shortfall'] = es_results
        
        # Calculate maximum drawdown
        drawdown_results = self.calculate_maximum_drawdown(returns)
        analysis_results['drawdown'] = drawdown_results
        
        # Detect yield curve inversion
        inversion_results = self.detect_yield_curve_inversion(data)
        analysis_results['inversion'] = inversion_results
        
        # Perform stress tests
        stress_results = self.stress_test_yield_curve(data)
        analysis_results['stress_tests'] = stress_results
        
        # Monte Carlo simulation
        mc_results = self.monte_carlo_simulation(data)
        analysis_results['monte_carlo'] = mc_results
        
        return analysis_results
