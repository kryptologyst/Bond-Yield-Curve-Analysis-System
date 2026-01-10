"""Training and evaluation scripts for Bond Yield Curve Analysis System."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
from omegaconf import OmegaConf

from src.data.loader import TreasuryDataLoader
from src.models.forecasting import YieldCurveForecaster
from src.risk.analyzer import YieldCurveRiskAnalyzer
from src.utils.config import SystemConfig
from src.utils.utils import setup_logging, set_random_seeds


def train_model(config_path: str, output_dir: str = "assets") -> Dict[str, Any]:
    """Train the yield curve forecasting model.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing training results
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting model training")
    
    # Load configuration
    config = SystemConfig.from_yaml(config_path)
    set_random_seeds(config.random_seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading data")
    data_loader = TreasuryDataLoader(config.data)
    data = data_loader.load_treasury_data()
    data = data_loader.preprocess_data(data)
    
    # Create train-test split
    logger.info("Creating train-test split")
    train_data, val_data, test_data = data_loader.create_train_test_split(
        data,
        test_size=config.model.test_split,
        validation_size=config.model.validation_split
    )
    
    # Train model
    logger.info(f"Training {config.model.type} model")
    forecaster = YieldCurveForecaster(config.model)
    forecaster.fit(train_data)
    
    # Make predictions
    logger.info("Making predictions")
    train_predictions = forecaster.predict(horizon=config.model.horizon)
    val_predictions = forecaster.predict(horizon=config.model.horizon)
    test_predictions = forecaster.predict(horizon=config.model.horizon)
    
    # Evaluate model
    logger.info("Evaluating model")
    evaluation_results = {}
    
    # Save results
    logger.info("Saving results")
    results = {
        'config': config.to_dict(),
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'train_predictions': train_predictions,
        'val_predictions': val_predictions,
        'test_predictions': test_predictions,
        'model_info': forecaster.get_model_info(),
        'evaluation': evaluation_results
    }
    
    # Save data
    train_data.to_csv(output_path / 'train_data.csv', index=False)
    val_data.to_csv(output_path / 'val_data.csv', index=False)
    test_data.to_csv(output_path / 'test_data.csv', index=False)
    
    # Save predictions
    pd.DataFrame(train_predictions['predictions']).to_csv(output_path / 'train_predictions.csv')
    pd.DataFrame(val_predictions['predictions']).to_csv(output_path / 'val_predictions.csv')
    pd.DataFrame(test_predictions['predictions']).to_csv(output_path / 'test_predictions.csv')
    
    logger.info(f"Training completed. Results saved to {output_path}")
    return results


def evaluate_model(config_path: str, model_path: str = None) -> Dict[str, Any]:
    """Evaluate the trained model.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to saved model (optional)
        
    Returns:
        Dictionary containing evaluation results
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting model evaluation")
    
    # Load configuration
    config = SystemConfig.from_yaml(config_path)
    set_random_seeds(config.random_seed)
    
    # Load data
    logger.info("Loading data")
    data_loader = TreasuryDataLoader(config.data)
    data = data_loader.load_treasury_data()
    data = data_loader.preprocess_data(data)
    
    # Create train-test split
    train_data, val_data, test_data = data_loader.create_train_test_split(
        data,
        test_size=config.model.test_split,
        validation_size=config.model.validation_split
    )
    
    # Train model (or load if path provided)
    logger.info("Training/loading model")
    forecaster = YieldCurveForecaster(config.model)
    forecaster.fit(train_data)
    
    # Make predictions on test set
    logger.info("Making test predictions")
    test_predictions = forecaster.predict(horizon=config.model.horizon)
    
    # Calculate evaluation metrics
    logger.info("Calculating evaluation metrics")
    evaluation_results = {}
    
    # For each symbol, calculate metrics
    for symbol in test_data['symbol'].unique():
        symbol_test_data = test_data[test_data['symbol'] == symbol].sort_values('date')
        
        if symbol in test_predictions['predictions']:
            # Get actual values (last values in test set)
            actual_values = symbol_test_data['yield'].tail(config.model.horizon).values
            
            # Get predicted values
            predicted_values = test_predictions['predictions'][symbol]
            
            # Calculate metrics
            mse = np.mean((actual_values - predicted_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual_values - predicted_values))
            mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
            
            evaluation_results[symbol] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
    
    logger.info("Evaluation completed")
    return evaluation_results


def run_backtest(config_path: str, output_dir: str = "assets") -> Dict[str, Any]:
    """Run backtesting on the model.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing backtest results
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting backtesting")
    
    # Load configuration
    config = SystemConfig.from_yaml(config_path)
    set_random_seeds(config.random_seed)
    
    # Load data
    logger.info("Loading data")
    data_loader = TreasuryDataLoader(config.data)
    data = data_loader.load_treasury_data()
    data = data_loader.preprocess_data(data)
    
    # Create train-test split
    train_data, val_data, test_data = data_loader.create_train_test_split(
        data,
        test_size=config.model.test_split,
        validation_size=config.model.validation_split
    )
    
    # Train model
    logger.info("Training model for backtesting")
    forecaster = YieldCurveForecaster(config.model)
    forecaster.fit(train_data)
    
    # Run backtest
    logger.info("Running backtest")
    backtest_results = {
        'initial_capital': config.backtest.initial_capital,
        'transaction_cost_bps': config.backtest.transaction_cost_bps,
        'slippage_bps': config.backtest.slippage_bps,
        'returns': [],
        'positions': [],
        'trades': []
    }
    
    # Simple backtest: buy/sell based on yield curve slope
    current_capital = config.backtest.initial_capital
    positions = {}
    
    for date in test_data['date'].unique():
        date_data = test_data[test_data['date'] == date]
        
        # Calculate yield curve slope
        slope = (date_data[date_data['symbol'] == 'DGS10']['yield'].iloc[0] - 
                date_data[date_data['symbol'] == 'DGS2']['yield'].iloc[0])
        
        # Simple strategy: buy long-term bonds when slope is high, short-term when low
        if slope > 1.0:  # Steep curve
            # Buy 10Y bonds
            if 'DGS10' not in positions:
                positions['DGS10'] = 0.5  # 50% allocation
                backtest_results['trades'].append({
                    'date': date,
                    'action': 'buy',
                    'symbol': 'DGS10',
                    'allocation': 0.5
                })
        elif slope < -0.5:  # Inverted curve
            # Buy 2Y bonds
            if 'DGS2' not in positions:
                positions['DGS2'] = 0.5  # 50% allocation
                backtest_results['trades'].append({
                    'date': date,
                    'action': 'buy',
                    'symbol': 'DGS2',
                    'allocation': 0.5
                })
        
        # Calculate portfolio return
        portfolio_return = 0
        for symbol, allocation in positions.items():
            symbol_data = date_data[date_data['symbol'] == symbol]
            if not symbol_data.empty:
                symbol_return = symbol_data['yield'].iloc[0] / 100  # Convert to decimal
                portfolio_return += allocation * symbol_return
        
        current_capital *= (1 + portfolio_return)
        backtest_results['returns'].append(portfolio_return)
        backtest_results['positions'].append(positions.copy())
    
    # Calculate final metrics
    total_return = (current_capital - config.backtest.initial_capital) / config.backtest.initial_capital
    annualized_return = (1 + total_return) ** (252 / len(backtest_results['returns'])) - 1
    volatility = np.std(backtest_results['returns']) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    backtest_results['final_capital'] = current_capital
    backtest_results['total_return'] = total_return
    backtest_results['annualized_return'] = annualized_return
    backtest_results['volatility'] = volatility
    backtest_results['sharpe_ratio'] = sharpe_ratio
    
    logger.info(f"Backtest completed. Final capital: ${current_capital:,.2f}")
    logger.info(f"Total return: {total_return:.2%}")
    logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
    
    return backtest_results


def main():
    """Main function for training and evaluation scripts."""
    parser = argparse.ArgumentParser(description="Bond Yield Curve Analysis - Training and Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="assets", 
                       help="Output directory for results")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "backtest"], 
                       default="train", help="Mode to run")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BOND YIELD CURVE ANALYSIS - TRAINING & EVALUATION")
    print("=" * 80)
    print("DISCLAIMER: This is for research and educational purposes only.")
    print("NOT investment advice. Use at your own risk.")
    print("=" * 80)
    
    try:
        if args.mode == "train":
            results = train_model(args.config, args.output_dir)
            print(f"\nTraining completed successfully!")
            print(f"Results saved to: {args.output_dir}")
            
        elif args.mode == "evaluate":
            results = evaluate_model(args.config)
            print(f"\nEvaluation completed successfully!")
            print("Evaluation Results:")
            for symbol, metrics in results.items():
                print(f"  {symbol}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
                
        elif args.mode == "backtest":
            results = run_backtest(args.config, args.output_dir)
            print(f"\nBacktest completed successfully!")
            print(f"Final capital: ${results['final_capital']:,.2f}")
            print(f"Total return: {results['total_return']:.2%}")
            print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        
        print("\n" + "=" * 80)
        print("OPERATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during {args.mode}: {e}")
        logging.error(f"{args.mode} failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
