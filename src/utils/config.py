"""Configuration management for Bond Yield Curve Analysis System."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml
from omegaconf import DictConfig, OmegaConf


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    source: str = "fred"  # "fred", "synthetic", "csv"
    symbols: List[str] = None
    start_date: str = "2000-01-01"
    end_date: str = "2024-01-01"
    frequency: str = "daily"
    api_key: Optional[str] = None
    
    def __post_init__(self) -> None:
        if self.symbols is None:
            self.symbols = [
                "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", 
                "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"
            ]


@dataclass
class ModelConfig:
    """Configuration for forecasting models."""
    
    type: str = "var"  # "arima", "var", "regime_switching", "neural_network"
    horizon: int = 30
    confidence_levels: List[float] = None
    validation_split: float = 0.2
    test_split: float = 0.1
    
    def __post_init__(self) -> None:
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99]


@dataclass
class RiskConfig:
    """Configuration for risk analysis."""
    
    var_horizon: int = 1
    var_confidence_levels: List[float] = None
    stress_scenarios: List[str] = None
    inversion_threshold: float = -0.5  # basis points
    max_drawdown_threshold: float = 0.15
    
    def __post_init__(self) -> None:
        if self.var_confidence_levels is None:
            self.var_confidence_levels = [0.95, 0.99]
        if self.stress_scenarios is None:
            self.stress_scenarios = ["recession", "inflation_shock", "flight_to_quality"]


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    initial_capital: float = 100000.0
    transaction_cost_bps: float = 5.0  # basis points
    slippage_bps: float = 2.0
    rebalance_frequency: str = "monthly"
    benchmark_symbol: str = "DGS10"


@dataclass
class SystemConfig:
    """Main system configuration."""
    
    data: DataConfig
    model: ModelConfig
    risk: RiskConfig
    backtest: BacktestConfig
    
    # System settings
    random_seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    log_level: str = "INFO"
    output_dir: str = "assets"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> SystemConfig:
        """Load configuration from YAML file."""
        config_dict = OmegaConf.load(config_path)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> SystemConfig:
        """Create configuration from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            risk=RiskConfig(**config_dict.get("risk", {})),
            backtest=BacktestConfig(**config_dict.get("backtest", {})),
            random_seed=config_dict.get("random_seed", 42),
            device=config_dict.get("device", "auto"),
            log_level=config_dict.get("log_level", "INFO"),
            output_dir=config_dict.get("output_dir", "assets"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "risk": self.risk.__dict__,
            "backtest": self.backtest.__dict__,
            "random_seed": self.random_seed,
            "device": self.device,
            "log_level": self.log_level,
            "output_dir": self.output_dir,
        }
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def get_default_config() -> SystemConfig:
    """Get default system configuration."""
    return SystemConfig(
        data=DataConfig(),
        model=ModelConfig(),
        risk=RiskConfig(),
        backtest=BacktestConfig(),
    )
