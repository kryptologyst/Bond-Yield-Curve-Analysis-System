"""Modernized Bond Yield Curve Analysis System - Main Script."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

from src.data.loader import TreasuryDataLoader
from src.models.forecasting import YieldCurveForecaster
from src.risk.analyzer import YieldCurveRiskAnalyzer
from src.utils.config import SystemConfig
from src.utils.utils import (
    setup_logging, 
    set_random_seeds, 
    get_device,
    calculate_yield_curve_metrics,
    detect_yield_curve_inversion
)


class BondYieldCurveAnalysis:
    """Main class for Bond Yield Curve Analysis System."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the analysis system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = SystemConfig.from_yaml(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config.log_level)
        
        # Set random seeds for reproducibility
        set_random_seeds(self.config.random_seed)
        
        # Setup device
        self.device = get_device(self.config.device)
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = TreasuryDataLoader(self.config.data)
        self.forecaster = YieldCurveForecaster(self.config.model)
        self.risk_analyzer = YieldCurveRiskAnalyzer(self.config.risk)
        
        self.logger.info("Bond Yield Curve Analysis System initialized")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run the complete yield curve analysis.
        
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Starting Bond Yield Curve Analysis")
        
        results = {}
        
        # Step 1: Load and preprocess data
        self.logger.info("Step 1: Loading and preprocessing data")
        data = self.data_loader.load_treasury_data()
        data = self.data_loader.preprocess_data(data)
        results['data'] = data
        
        # Step 2: Create labels
        self.logger.info("Step 2: Creating forward-looking labels")
        labels = self.data_loader.create_labels(data)
        results['labels'] = labels
        
        # Step 3: Train-test split
        self.logger.info("Step 3: Creating train-test splits")
        train_data, val_data, test_data = self.data_loader.create_train_test_split(
            data, 
            test_size=self.config.model.test_split,
            validation_size=self.config.model.validation_split
        )
        results['splits'] = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        # Step 4: Train forecasting model
        self.logger.info("Step 4: Training forecasting model")
        self.forecaster.fit(train_data)
        results['model_info'] = self.forecaster.get_model_info()
        
        # Step 5: Make predictions
        self.logger.info("Step 5: Making predictions")
        predictions = self.forecaster.predict(horizon=self.config.model.horizon)
        results['predictions'] = predictions
        
        # Step 6: Risk analysis
        self.logger.info("Step 6: Performing risk analysis")
        risk_analysis = self.risk_analyzer.analyze(data)
        results['risk_analysis'] = risk_analysis
        
        # Step 7: Generate visualizations
        self.logger.info("Step 7: Generating visualizations")
        self._create_visualizations(data, predictions, risk_analysis)
        
        # Step 8: Save results
        self.logger.info("Step 8: Saving results")
        self._save_results(results)
        
        self.logger.info("Bond Yield Curve Analysis completed successfully")
        return results
    
    def _create_visualizations(
        self, 
        data: pd.DataFrame, 
        predictions: Dict[str, Any], 
        risk_analysis: Dict[str, Any]
    ) -> None:
        """Create comprehensive visualizations."""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Yield Curve Evolution
        self._plot_yield_curve_evolution(data)
        
        # 2. Forecasting Results
        self._plot_forecasting_results(data, predictions)
        
        # 3. Risk Analysis
        self._plot_risk_analysis(risk_analysis)
        
        # 4. Yield Curve Shape Analysis
        self._plot_yield_curve_shapes(data)
        
        # 5. Inversion Detection
        self._plot_inversion_analysis(data, risk_analysis)
    
    def _plot_yield_curve_evolution(self, data: pd.DataFrame) -> None:
        """Plot yield curve evolution over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Yield Curve Evolution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Yield curves over time
        ax1 = axes[0, 0]
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_values('date')
            ax1.plot(symbol_data['date'], symbol_data['yield'], 
                    label=symbol, alpha=0.7, linewidth=1)
        ax1.set_title('Yield Curves Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Yield (%)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Current yield curve
        ax2 = axes[0, 1]
        latest_data = data[data['date'] == data['date'].max()]
        ax2.plot(latest_data['maturity'], latest_data['yield'], 
                'o-', linewidth=2, markersize=8)
        ax2.set_title('Current Yield Curve')
        ax2.set_xlabel('Maturity (Years)')
        ax2.set_ylabel('Yield (%)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Yield curve slope over time
        ax3 = axes[1, 0]
        slope_data = []
        for date in data['date'].unique():
            date_data = data[data['date'] == date]
            metrics = calculate_yield_curve_metrics(
                date_data['yield'], 
                date_data['maturity']
            )
            slope_data.append({'date': date, 'slope': metrics['slope']})
        
        slope_df = pd.DataFrame(slope_data)
        ax3.plot(slope_df['date'], slope_df['slope'], linewidth=2)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax3.set_title('Yield Curve Slope Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Slope (10Y - 2Y)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Yield volatility
        ax4 = axes[1, 1]
        volatility_data = []
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_values('date')
            volatility = symbol_data['yield'].rolling(22).std().iloc[-1]
            volatility_data.append({'symbol': symbol, 'volatility': volatility})
        
        vol_df = pd.DataFrame(volatility_data)
        ax4.bar(vol_df['symbol'], vol_df['volatility'])
        ax4.set_title('Yield Volatility by Maturity')
        ax4.set_xlabel('Maturity')
        ax4.set_ylabel('Volatility')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'yield_curve_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_forecasting_results(self, data: pd.DataFrame, predictions: Dict[str, Any]) -> None:
        """Plot forecasting results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Yield Curve Forecasting Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Forecasted yield curves
        ax1 = axes[0, 0]
        latest_data = data[data['date'] == data['date'].max()]
        
        # Plot current curve
        ax1.plot(latest_data['maturity'], latest_data['yield'], 
                'o-', label='Current', linewidth=2, markersize=8)
        
        # Plot forecasted curves
        if 'predictions' in predictions:
            pred_data = predictions['predictions']
            for symbol, forecast in pred_data.items():
                if symbol in latest_data['symbol'].values:
                    maturity = latest_data[latest_data['symbol'] == symbol]['maturity'].iloc[0]
                    ax1.plot([maturity], [forecast[-1]], 's', 
                           label=f'Forecast {symbol}', markersize=8)
        
        ax1.set_title('Yield Curve Forecast')
        ax1.set_xlabel('Maturity (Years)')
        ax1.set_ylabel('Yield (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Forecast confidence intervals
        ax2 = axes[0, 1]
        if 'confidence_intervals' in predictions:
            ci_data = predictions['confidence_intervals']
            for symbol, ci in ci_data.items():
                if symbol in latest_data['symbol'].values:
                    maturity = latest_data[latest_data['symbol'] == symbol]['maturity'].iloc[0]
                    current_yield = latest_data[latest_data['symbol'] == symbol]['yield'].iloc[0]
                    
                    # Plot confidence interval
                    ax2.fill_between([maturity-0.1, maturity+0.1], 
                                   [ci[-1, 0], ci[-1, 0]], 
                                   [ci[-1, 1], ci[-1, 1]], 
                                   alpha=0.3, label=f'{symbol} CI')
        
        ax2.set_title('Forecast Confidence Intervals')
        ax2.set_xlabel('Maturity (Years)')
        ax2.set_ylabel('Yield (%)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Forecast horizon
        ax3 = axes[1, 0]
        horizon = predictions.get('horizon', 30)
        forecast_dates = pd.date_range(
            start=data['date'].max(), 
            periods=horizon+1, 
            freq='D'
        )
        
        for symbol in ['DGS2', 'DGS10']:  # Focus on key maturities
            if symbol in predictions['predictions']:
                forecast_values = predictions['predictions'][symbol]
                ax3.plot(forecast_dates[1:], forecast_values, 
                        label=f'{symbol} Forecast', linewidth=2)
        
        ax3.set_title('Forecast Horizon')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Yield (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Forecast accuracy (if available)
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'Forecast Accuracy\n(Requires validation data)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Forecast Accuracy')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'forecasting_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_analysis(self, risk_analysis: Dict[str, Any]) -> None:
        """Plot risk analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Risk Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: VaR and ES
        ax1 = axes[0, 0]
        if 'var' in risk_analysis:
            var_data = risk_analysis['var']
            es_data = risk_analysis['expected_shortfall']
            
            metrics = ['var_95', 'var_99', 'es_95', 'es_99']
            values = [var_data.get('var_95', 0), var_data.get('var_99', 0),
                     es_data.get('es_95', 0), es_data.get('es_99', 0)]
            
            ax1.bar(metrics, values)
            ax1.set_title('Value at Risk and Expected Shortfall')
            ax1.set_ylabel('Risk Level')
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Drawdown analysis
        ax2 = axes[0, 1]
        if 'drawdown' in risk_analysis:
            drawdown_data = risk_analysis['drawdown']
            ax2.bar(['Max Drawdown'], [drawdown_data.get('max_drawdown', 0)])
            ax2.set_title('Maximum Drawdown')
            ax2.set_ylabel('Drawdown Level')
        
        # Plot 3: Stress test results
        ax3 = axes[1, 0]
        if 'stress_tests' in risk_analysis:
            stress_data = risk_analysis['stress_tests']
            scenarios = list(stress_data.keys())
            impacts = [stress_data[scenario].get('impact', 0) for scenario in scenarios]
            
            ax3.bar(scenarios, impacts)
            ax3.set_title('Stress Test Results')
            ax3.set_ylabel('Impact (bps)')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Inversion analysis
        ax4 = axes[1, 1]
        if 'inversion' in risk_analysis:
            inversion_data = risk_analysis['inversion']
            is_inverted = inversion_data.get('is_inverted', False)
            severity = inversion_data.get('severity', 0)
            
            ax4.bar(['Inversion Status'], [1 if is_inverted else 0])
            ax4.set_title('Yield Curve Inversion')
            ax4.set_ylabel('Inverted (1) / Normal (0)')
            ax4.set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_yield_curve_shapes(self, data: pd.DataFrame) -> None:
        """Plot different yield curve shapes."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Yield Curve Shape Analysis', fontsize=16, fontweight='bold')
        
        # Analyze curve shapes over time
        shape_analysis = []
        for date in data['date'].unique():
            date_data = data[data['date'] == date]
            metrics = calculate_yield_curve_metrics(
                date_data['yield'], 
                date_data['maturity']
            )
            shape_analysis.append({
                'date': date,
                'slope': metrics['slope'],
                'curvature': metrics['curvature'],
                'level': metrics['level'],
                'inverted': metrics['inverted']
            })
        
        shape_df = pd.DataFrame(shape_analysis)
        
        # Plot 1: Slope distribution
        ax1 = axes[0, 0]
        ax1.hist(shape_df['slope'], bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax1.set_title('Yield Curve Slope Distribution')
        ax1.set_xlabel('Slope (10Y - 2Y)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Curvature distribution
        ax2 = axes[0, 1]
        ax2.hist(shape_df['curvature'], bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_title('Yield Curve Curvature Distribution')
        ax2.set_xlabel('Curvature')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Slope vs Curvature
        ax3 = axes[1, 0]
        colors = ['red' if inv else 'blue' for inv in shape_df['inverted']]
        ax3.scatter(shape_df['slope'], shape_df['curvature'], c=colors, alpha=0.6)
        ax3.set_title('Slope vs Curvature')
        ax3.set_xlabel('Slope')
        ax3.set_ylabel('Curvature')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Inversion frequency over time
        ax4 = axes[1, 1]
        inversion_rate = shape_df['inverted'].rolling(252).mean()  # Annual rolling average
        ax4.plot(shape_df['date'], inversion_rate, linewidth=2)
        ax4.set_title('Yield Curve Inversion Rate')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Inversion Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'yield_curve_shapes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_inversion_analysis(self, data: pd.DataFrame, risk_analysis: Dict[str, Any]) -> None:
        """Plot yield curve inversion analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Yield Curve Inversion Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Inversion detection over time
        ax1 = axes[0, 0]
        inversion_data = []
        for date in data['date'].unique():
            date_data = data[data['date'] == date]
            is_inverted = detect_yield_curve_inversion(
                date_data['yield'], 
                date_data['maturity']
            )
            inversion_data.append({'date': date, 'inverted': is_inverted})
        
        inv_df = pd.DataFrame(inversion_data)
        ax1.fill_between(inv_df['date'], inv_df['inverted'], alpha=0.3, color='red')
        ax1.set_title('Yield Curve Inversion Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Inverted (1) / Normal (0)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Inversion severity
        ax2 = axes[0, 1]
        if 'inversion' in risk_analysis:
            inversion_info = risk_analysis['inversion']
            severity = inversion_info.get('severity', 0)
            severity_level = inversion_info.get('severity_level', 'normal')
            
            ax2.bar(['Current Severity'], [severity])
            ax2.set_title(f'Inversion Severity: {severity_level.title()}')
            ax2.set_ylabel('Severity Level')
        
        # Plot 3: Historical inversion periods
        ax3 = axes[1, 0]
        # Find inversion periods
        inv_periods = []
        in_period = False
        start_date = None
        
        for _, row in inv_df.iterrows():
            if row['inverted'] and not in_period:
                start_date = row['date']
                in_period = True
            elif not row['inverted'] and in_period:
                inv_periods.append((start_date, row['date']))
                in_period = False
        
        if in_period:  # Handle case where inversion continues to end
            inv_periods.append((start_date, inv_df['date'].iloc[-1]))
        
        # Plot inversion periods
        for start, end in inv_periods:
            ax3.axvspan(start, end, alpha=0.3, color='red')
        
        ax3.set_title('Historical Inversion Periods')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Inversion Status')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Inversion statistics
        ax4 = axes[1, 1]
        total_days = len(inv_df)
        inversion_days = inv_df['inverted'].sum()
        inversion_rate = inversion_days / total_days * 100
        
        stats_text = f"""
        Total Days: {total_days:,}
        Inversion Days: {inversion_days:,}
        Inversion Rate: {inversion_rate:.1f}%
        Inversion Periods: {len(inv_periods)}
        """
        
        ax4.text(0.5, 0.5, stats_text, ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax4.set_title('Inversion Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'inversion_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results."""
        # Save data
        results['data'].to_csv(self.output_dir / 'yield_curve_data.csv', index=False)
        results['labels'].to_csv(self.output_dir / 'labels.csv', index=False)
        
        # Save splits
        results['splits']['train'].to_csv(self.output_dir / 'train_data.csv', index=False)
        results['splits']['validation'].to_csv(self.output_dir / 'validation_data.csv', index=False)
        results['splits']['test'].to_csv(self.output_dir / 'test_data.csv', index=False)
        
        # Save predictions
        if 'predictions' in results:
            pred_df = pd.DataFrame(results['predictions']['predictions'])
            pred_df.to_csv(self.output_dir / 'predictions.csv')
        
        # Save risk analysis summary
        risk_summary = {
            'timestamp': datetime.now().isoformat(),
            'model_type': results['model_info']['model_type'],
            'risk_metrics': results['risk_analysis'].get('risk_metrics', {}),
            'var': results['risk_analysis'].get('var', {}),
            'inversion': results['risk_analysis'].get('inversion', {})
        }
        
        import json
        with open(self.output_dir / 'risk_summary.json', 'w') as f:
            json.dump(risk_summary, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {self.output_dir}")


def main():
    """Main function to run the Bond Yield Curve Analysis."""
    print("=" * 80)
    print("BOND YIELD CURVE ANALYSIS SYSTEM")
    print("=" * 80)
    print("DISCLAIMER: This is for research and educational purposes only.")
    print("NOT investment advice. Use at your own risk.")
    print("=" * 80)
    
    try:
        # Initialize analysis system
        analysis = BondYieldCurveAnalysis()
        
        # Run analysis
        results = analysis.run_analysis()
        
        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"Data loaded: {len(results['data'])} observations")
        print(f"Model type: {results['model_info']['model_type']}")
        print(f"Forecast horizon: {analysis.config.model.horizon} days")
        
        if 'inversion' in results['risk_analysis']:
            inversion_info = results['risk_analysis']['inversion']
            print(f"Yield curve inverted: {inversion_info.get('is_inverted', False)}")
            print(f"Inversion severity: {inversion_info.get('severity_level', 'normal')}")
        
        print(f"\nResults saved to: {analysis.output_dir}")
        print("Visualizations created:")
        print("- yield_curve_evolution.png")
        print("- forecasting_results.png") 
        print("- risk_analysis.png")
        print("- yield_curve_shapes.png")
        print("- inversion_analysis.png")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        logging.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
