"""Streamlit demo application for Bond Yield Curve Analysis System."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Bond Yield Curve Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add disclaimer banner
st.error("""
**DISCLAIMER**: This is a research and educational demonstration only. 
This software is NOT investment advice and should NOT be used for actual trading or investment decisions.
""")

# Title and description
st.title("📈 Bond Yield Curve Analysis System")
st.markdown("""
This interactive demo provides comprehensive analysis of bond yield curves, including:
- **Yield Curve Forecasting**: Predict future yield curve shapes using time series models
- **Risk Assessment**: VaR, stress testing, and yield curve inversion detection  
- **Economic Indicators**: Analyze yield curve shapes for economic insights
- **Portfolio Optimization**: Use yield curve information for bond portfolio construction
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source",
    ["synthetic", "fred"],
    help="Choose data source for analysis"
)

# Model selection
model_type = st.sidebar.selectbox(
    "Forecasting Model",
    ["var", "arima", "regime_switching"],
    help="Select the forecasting model to use"
)

# Forecast horizon
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=90,
    value=30,
    help="Number of days to forecast ahead"
)

# Analysis parameters
st.sidebar.header("Analysis Parameters")

# Risk parameters
var_confidence = st.sidebar.selectbox(
    "VaR Confidence Level",
    [0.95, 0.99],
    format_func=lambda x: f"{int(x*100)}%",
    help="Confidence level for Value at Risk calculation"
)

inversion_threshold = st.sidebar.slider(
    "Inversion Threshold (bps)",
    min_value=-2.0,
    max_value=0.0,
    value=-0.5,
    step=0.1,
    help="Threshold for detecting yield curve inversion"
)

# Main content
if st.button("🚀 Run Analysis", type="primary"):
    
    with st.spinner("Running Bond Yield Curve Analysis..."):
        
        # Import here to avoid issues with streamlit
        try:
            from main import BondYieldCurveAnalysis
            
            # Create temporary config
            config_dict = {
                "data": {
                    "source": data_source,
                    "symbols": ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"],
                    "start_date": "2020-01-01",
                    "end_date": "2024-01-01"
                },
                "model": {
                    "type": model_type,
                    "horizon": forecast_horizon,
                    "confidence_levels": [var_confidence, 0.99]
                },
                "risk": {
                    "inversion_threshold": inversion_threshold,
                    "var_confidence_levels": [var_confidence, 0.99]
                },
                "backtest": {},
                "random_seed": 42,
                "device": "auto",
                "log_level": "INFO",
                "output_dir": "assets"
            }
            
            # Save temporary config
            import yaml
            with open("temp_config.yaml", "w") as f:
                yaml.dump(config_dict, f)
            
            # Run analysis
            analysis = BondYieldCurveAnalysis("temp_config.yaml")
            results = analysis.run_analysis()
            
            # Clean up
            Path("temp_config.yaml").unlink(missing_ok=True)
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()
    
    # Display results
    st.success("✅ Analysis completed successfully!")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Yield Curve Overview", 
        "🔮 Forecasting Results", 
        "⚠️ Risk Analysis", 
        "📈 Shape Analysis", 
        "📋 Summary"
    ])
    
    with tab1:
        st.header("Yield Curve Overview")
        
        # Current yield curve
        latest_data = results['data'][results['data']['date'] == results['data']['date'].max()]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=latest_data['maturity'],
            y=latest_data['yield'],
            mode='lines+markers',
            name='Current Yield Curve',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Current Yield Curve",
            xaxis_title="Maturity (Years)",
            yaxis_title="Yield (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Yield curve metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Curve Level", f"{latest_data['yield'].mean():.2f}%")
        
        with col2:
            slope = latest_data[latest_data['symbol'] == 'DGS10']['yield'].iloc[0] - latest_data[latest_data['symbol'] == 'DGS2']['yield'].iloc[0]
            st.metric("Curve Slope", f"{slope:.2f}%")
        
        with col3:
            st.metric("Curve Volatility", f"{latest_data['yield'].std():.2f}%")
        
        with col4:
            is_inverted = slope < inversion_threshold
            st.metric("Inversion Status", "🔴 Inverted" if is_inverted else "🟢 Normal")
    
    with tab2:
        st.header("Forecasting Results")
        
        if 'predictions' in results:
            predictions = results['predictions']
            
            # Forecast comparison
            latest_data = results['data'][results['data']['date'] == results['data']['date'].max()]
            
            fig = go.Figure()
            
            # Current curve
            fig.add_trace(go.Scatter(
                x=latest_data['maturity'],
                y=latest_data['yield'],
                mode='lines+markers',
                name='Current',
                line=dict(width=3, color='blue'),
                marker=dict(size=8)
            ))
            
            # Forecasted curve
            if 'predictions' in predictions:
                forecast_maturities = []
                forecast_yields = []
                
                for symbol in ['DGS2', 'DGS5', 'DGS10', 'DGS30']:
                    if symbol in predictions['predictions']:
                        maturity = latest_data[latest_data['symbol'] == symbol]['maturity'].iloc[0]
                        forecast_yield = predictions['predictions'][symbol][-1]
                        forecast_maturities.append(maturity)
                        forecast_yields.append(forecast_yield)
                
                if forecast_maturities:
                    fig.add_trace(go.Scatter(
                        x=forecast_maturities,
                        y=forecast_yields,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(width=3, color='red', dash='dash'),
                        marker=dict(size=8)
                    ))
            
            fig.update_layout(
                title=f"Yield Curve Forecast ({forecast_horizon} days ahead)",
                xaxis_title="Maturity (Years)",
                yaxis_title="Yield (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast horizon chart
            if 'predictions' in predictions:
                forecast_dates = pd.date_range(
                    start=results['data']['date'].max(),
                    periods=forecast_horizon+1,
                    freq='D'
                )[1:]
                
                fig2 = go.Figure()
                
                for symbol in ['DGS2', 'DGS10']:
                    if symbol in predictions['predictions']:
                        fig2.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=predictions['predictions'][symbol],
                            mode='lines',
                            name=f'{symbol} Forecast',
                            line=dict(width=2)
                        ))
                
                fig2.update_layout(
                    title="Forecast Horizon",
                    xaxis_title="Date",
                    yaxis_title="Yield (%)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        else:
            st.warning("No forecasting results available")
    
    with tab3:
        st.header("Risk Analysis")
        
        if 'risk_analysis' in results:
            risk_data = results['risk_analysis']
            
            # Risk metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Value at Risk")
                if 'var' in risk_data:
                    var_data = risk_data['var']
                    for key, value in var_data.items():
                        st.metric(key.replace('_', ' ').title(), f"{value:.3f}")
            
            with col2:
                st.subheader("Expected Shortfall")
                if 'expected_shortfall' in risk_data:
                    es_data = risk_data['expected_shortfall']
                    for key, value in es_data.items():
                        st.metric(key.replace('_', ' ').title(), f"{value:.3f}")
            
            # Drawdown analysis
            st.subheader("Drawdown Analysis")
            if 'drawdown' in risk_data:
                drawdown_data = risk_data['drawdown']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Max Drawdown", f"{drawdown_data.get('max_drawdown', 0):.3f}")
                
                with col2:
                    duration = drawdown_data.get('drawdown_duration', 0)
                    st.metric("Drawdown Duration", f"{duration} days" if duration else "N/A")
                
                with col3:
                    st.metric("Recovery Status", "✅ Recovered" if duration else "🔄 Ongoing")
            
            # Stress test results
            st.subheader("Stress Test Results")
            if 'stress_tests' in risk_data:
                stress_data = risk_data['stress_tests']
                
                stress_df = pd.DataFrame([
                    {
                        'Scenario': scenario,
                        'Impact': data.get('impact', 0),
                        'Current': data.get('current_level', data.get('current_slope', 0)),
                        'Stressed': data.get('stressed_level', data.get('stressed_slope', 0))
                    }
                    for scenario, data in stress_data.items()
                ])
                
                st.dataframe(stress_df, use_container_width=True)
                
                # Stress test visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=stress_df['Scenario'],
                    y=stress_df['Impact'],
                    name='Impact',
                    marker_color=['red' if x < 0 else 'green' for x in stress_df['Impact']]
                ))
                
                fig.update_layout(
                    title="Stress Test Impact",
                    xaxis_title="Scenario",
                    yaxis_title="Impact (bps)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No risk analysis results available")
    
    with tab4:
        st.header("Yield Curve Shape Analysis")
        
        # Calculate shape metrics over time
        shape_metrics = []
        for date in results['data']['date'].unique():
            date_data = results['data'][results['data']['date'] == date]
            if len(date_data) >= 2:
                slope = date_data[date_data['symbol'] == 'DGS10']['yield'].iloc[0] - date_data[date_data['symbol'] == 'DGS2']['yield'].iloc[0]
                level = date_data['yield'].mean()
                volatility = date_data['yield'].std()
                
                shape_metrics.append({
                    'date': date,
                    'slope': slope,
                    'level': level,
                    'volatility': volatility,
                    'inverted': slope < inversion_threshold
                })
        
        shape_df = pd.DataFrame(shape_metrics)
        
        # Slope over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=shape_df['date'],
            y=shape_df['slope'],
            mode='lines',
            name='Slope',
            line=dict(width=2)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)
        fig.add_hline(y=inversion_threshold, line_dash="dash", line_color="orange", opacity=0.7)
        
        fig.update_layout(
            title="Yield Curve Slope Over Time",
            xaxis_title="Date",
            yaxis_title="Slope (10Y - 2Y)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Shape distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                shape_df, 
                x='slope', 
                title='Slope Distribution',
                nbins=30
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.add_vline(x=inversion_threshold, line_dash="dash", line_color="orange")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                shape_df, 
                x='volatility', 
                title='Volatility Distribution',
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Inversion analysis
        st.subheader("Inversion Analysis")
        
        inversion_rate = shape_df['inverted'].mean() * 100
        inversion_periods = (shape_df['inverted'].diff() == 1).sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Inversion Rate", f"{inversion_rate:.1f}%")
        
        with col2:
            st.metric("Inversion Periods", f"{inversion_periods}")
        
        with col3:
            current_inverted = shape_df['inverted'].iloc[-1]
            st.metric("Current Status", "🔴 Inverted" if current_inverted else "🟢 Normal")
    
    with tab5:
        st.header("Analysis Summary")
        
        # Model information
        st.subheader("Model Information")
        if 'model_info' in results:
            model_info = results['model_info']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", model_info.get('model_type', 'N/A').upper())
            
            with col2:
                st.metric("Forecast Horizon", f"{forecast_horizon} days")
            
            with col3:
                st.metric("Model Status", "✅ Fitted" if model_info.get('is_fitted', False) else "❌ Not Fitted")
        
        # Data summary
        st.subheader("Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Observations", f"{len(results['data']):,}")
        
        with col2:
            st.metric("Date Range", f"{(results['data']['date'].max() - results['data']['date'].min()).days} days")
        
        with col3:
            st.metric("Maturities", f"{results['data']['symbol'].nunique()}")
        
        with col4:
            st.metric("Data Source", data_source.title())
        
        # Key findings
        st.subheader("Key Findings")
        
        # Current yield curve status
        latest_data = results['data'][results['data']['date'] == results['data']['date'].max()]
        current_slope = latest_data[latest_data['symbol'] == 'DGS10']['yield'].iloc[0] - latest_data[latest_data['symbol'] == 'DGS2']['yield'].iloc[0]
        is_inverted = current_slope < inversion_threshold
        
        findings = [
            f"**Current Yield Curve**: {'Inverted' if is_inverted else 'Normal'} (slope: {current_slope:.2f}%)",
            f"**Curve Level**: {latest_data['yield'].mean():.2f}%",
            f"**Curve Volatility**: {latest_data['yield'].std():.2f}%",
            f"**Forecast Model**: {model_type.upper()}",
            f"**Risk Assessment**: VaR calculated at {int(var_confidence*100)}% confidence level"
        ]
        
        for finding in findings:
            st.markdown(finding)
        
        # Download results
        st.subheader("Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = results['data'].to_csv(index=False)
            st.download_button(
                label="📊 Download Data",
                data=csv_data,
                file_name=f"yield_curve_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if 'predictions' in results:
                pred_data = pd.DataFrame(results['predictions']['predictions'])
                csv_pred = pred_data.to_csv()
                st.download_button(
                    label="🔮 Download Predictions",
                    data=csv_pred,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            risk_summary = json.dumps(results.get('risk_analysis', {}), indent=2, default=str)
            st.download_button(
                label="⚠️ Download Risk Analysis",
                data=risk_summary,
                file_name=f"risk_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This application is for research and educational purposes only. 
It is not intended as investment advice. Always consult with qualified financial 
professionals before making investment decisions.
""")

# Add some styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)
