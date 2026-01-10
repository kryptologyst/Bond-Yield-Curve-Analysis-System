# Bond Yield Curve Analysis System

## DISCLAIMER

**IMPORTANT: This is a research and educational demonstration only. This software is NOT investment advice and should NOT be used for actual trading or investment decisions.**

- This system is for academic research and educational purposes only
- All results are hypothetical and may be inaccurate
- Backtests are simulations and do not represent actual trading results
- Past performance does not guarantee future results
- Always consult with qualified financial professionals before making investment decisions
- The authors assume no responsibility for any financial losses

## Overview

This project provides a comprehensive system for analyzing and forecasting bond yield curves, with applications in:

- **Yield Curve Forecasting**: Predicting future yield curve shapes using time series models
- **Risk Assessment**: VaR, stress testing, and yield curve inversion detection
- **Economic Indicators**: Analyzing yield curve shapes for economic insights
- **Portfolio Optimization**: Using yield curve information for bond portfolio construction

## Features

- **Data Sources**: FRED API integration for real Treasury data, synthetic data generation
- **Models**: ARIMA, VAR, regime switching, neural networks for yield curve forecasting
- **Risk Analysis**: VaR calculation, stress testing, yield curve inversion detection
- **Visualization**: Interactive plots, yield curve animations, risk dashboards
- **Backtesting**: Historical performance evaluation with transaction costs
- **Demo Interface**: Streamlit/Gradio web interface for interactive analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Bond-Yield-Curve-Analysis-System.git
cd Bond-Yield-Curve-Analysis-System

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For advanced features
pip install -e ".[advanced]"
```

### Basic Usage

```python
from src.data import TreasuryDataLoader
from src.models import YieldCurveForecaster
from src.risk import YieldCurveRiskAnalyzer

# Load data
loader = TreasuryDataLoader()
data = loader.load_treasury_data()

# Train model
forecaster = YieldCurveForecaster()
forecaster.fit(data)

# Make predictions
predictions = forecaster.predict(horizon=30)

# Risk analysis
risk_analyzer = YieldCurveRiskAnalyzer()
risk_metrics = risk_analyzer.analyze(data)
```

### Demo Interface

```bash
# Launch Streamlit demo
streamlit run demo/app.py

# Or launch Gradio demo
python demo/gradio_app.py
```

## Data Schema

### Market Data (`data/market_data.csv`)
- `date`: Trading date
- `symbol`: Bond identifier (e.g., 'DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30')
- `yield`: Yield rate (%)
- `maturity`: Time to maturity (years)

### Labels (`data/labels.csv`)
- `date`: Date
- `symbol`: Bond identifier
- `forward_return_1m`: 1-month forward return
- `forward_return_3m`: 3-month forward return
- `forward_return_6m`: 6-month forward return
- `yield_change_1m`: 1-month yield change
- `curve_slope`: 10Y-2Y spread
- `curve_inversion`: Binary indicator for inversion

## Model Performance

### Forecasting Metrics
- **RMSE**: Root Mean Square Error for yield predictions
- **MAE**: Mean Absolute Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error

### Risk Metrics
- **VaR**: Value at Risk (95%, 99% confidence levels)
- **Expected Shortfall**: Conditional VaR
- **Maximum Drawdown**: Peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns
- **Calmar Ratio**: Return to max drawdown ratio

### Trading Metrics
- **Hit Rate**: Percentage of correct directional predictions
- **Information Ratio**: Active return to tracking error
- **Turnover**: Portfolio turnover rate
- **Transaction Costs**: Impact of trading costs

## Configuration

The system uses Hydra/OmegaConf for configuration management:

```yaml
# configs/config.yaml
data:
  source: "fred"  # or "synthetic"
  symbols: ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
  start_date: "2000-01-01"
  end_date: "2024-01-01"

model:
  type: "var"  # arima, var, regime_switching, neural_network
  horizon: 30
  confidence_levels: [0.95, 0.99]

risk:
  var_horizon: 1
  stress_scenarios: ["recession", "inflation_shock", "flight_to_quality"]
  inversion_threshold: -0.5  # basis points
```

## Development

### Code Quality
- **Formatting**: Black for code formatting
- **Linting**: Ruff for code linting
- **Type Checking**: MyPy for static type checking
- **Testing**: Pytest for unit tests

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

### Running Tests
```bash
pytest tests/ -v
```

## File Structure

```
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # Forecasting models
│   ├── backtest/       # Backtesting framework
│   ├── risk/           # Risk analysis tools
│   └── utils/          # Utility functions
├── configs/            # Configuration files
├── scripts/            # Training and evaluation scripts
├── notebooks/          # Jupyter notebooks for analysis
├── tests/              # Unit tests
├── assets/             # Generated plots and results
├── demo/               # Demo applications
└── data/               # Data storage
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{bond_yield_curve_analysis,
  title={Bond Yield Curve Analysis System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Bond-Yield-Curve-Analysis-System}
}
```
# Bond-Yield-Curve-Analysis-System
