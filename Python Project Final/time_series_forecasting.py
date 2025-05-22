import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def load_data(file_path):
    """Load and prepare time series data"""
    print("Loading and preparing time series data...")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Aggregate sales by date
    daily_sales = df.groupby('date')['sales_amount'].sum().reset_index()
    daily_sales.set_index('date', inplace=True)
    
    return daily_sales

def check_stationarity(time_series):
    """Check if the time series is stationary using ADF test"""
    print("\nChecking stationarity of time series...")
    
    # Perform Augmented Dickey-Fuller test
    result = adfuller(time_series.values)
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    # Interpret the result
    if result[1] <= 0.05:
        print("The time series is stationary (reject H0)")
        return True
    else:
        print("The time series is not stationary (fail to reject H0)")
        return False

def plot_time_series_components(time_series):
    """Plot time series components and diagnostics"""
    print("\nPlotting time series components...")
    
    plt.figure(figsize=(15, 10))
    
    # Original Time Series
    plt.subplot(3, 1, 1)
    plt.plot(time_series)
    plt.title('Original Time Series')
    plt.xlabel('Date')
    plt.ylabel('Sales Amount')
    
    # ACF Plot
    plt.subplot(3, 1, 2)
    plot_acf(time_series, ax=plt.gca(), lags=30)
    plt.title('Autocorrelation Function')
    
    # PACF Plot
    plt.subplot(3, 1, 3)
    plot_pacf(time_series, ax=plt.gca(), lags=30)
    plt.title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.savefig('time_series_diagnostics.png')
    print("Time series diagnostics saved as 'time_series_diagnostics.png'")

def difference_series(time_series, order=1):
    """Apply differencing to make the series stationary"""
    print(f"\nApplying {order}-order differencing...")
    
    differenced = time_series.diff(order).dropna()
    
    # Plot the differenced series
    plt.figure(figsize=(12, 6))
    plt.plot(differenced)
    plt.title(f'{order}-Order Differenced Time Series')
    plt.xlabel('Date')
    plt.ylabel('Differenced Sales')
    plt.grid(True)
    plt.savefig(f'differenced_series_{order}.png')
    print(f"Differenced series saved as 'differenced_series_{order}.png'")
    
    return differenced

def build_arima_model(time_series, order):
    """Build and fit an ARIMA model"""
    print(f"\nBuilding ARIMA{order} model...")
    
    # Fit the ARIMA model
    model = ARIMA(time_series, order=order)
    model_fit = model.fit()
    
    # Print model summary
    print("\nARIMA Model Summary:")
    print(model_fit.summary())
    
    return model_fit

def forecast_with_arima(model, steps=30):
    """Generate forecasts using the ARIMA model"""
    print(f"\nForecasting {steps} steps ahead with ARIMA...")
    
    # Make forecast
    forecast = model.forecast(steps=steps)
    forecast_index = pd.date_range(start=model.data.index[-1] + pd.Timedelta(days=1), periods=steps)
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(model.data.index, model.data, label='Historical Data')
    plt.plot(forecast_series.index, forecast_series, 'r--', label='ARIMA Forecast')
    plt.title('ARIMA Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales Amount')
    plt.legend()
    plt.grid(True)
    plt.savefig('arima_forecast.png')
    print("ARIMA forecast saved as 'arima_forecast.png'")
    
    return forecast_series

def seasonal_decomposition(time_series):
    """Decompose time series into trend, seasonal, and residual components"""
    print("\nPerforming seasonal decomposition...")
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Ensure we have enough data for decomposition
    if len(time_series) < 14:  # Need at least 2 weeks for weekly seasonality
        print("Not enough data for seasonal decomposition")
        return None
    
    # Perform decomposition
    decomposition = seasonal_decompose(time_series, model='additive', period=7)  # Assuming weekly seasonality
    
    # Plot decomposition
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(time_series, label='Original')
    plt.legend(loc='upper left')
    
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='upper left')
    
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend(loc='upper left')
    
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label='Residuals')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('seasonal_decomposition.png')
    print("Seasonal decomposition saved as 'seasonal_decomposition.png'")
    
    return decomposition

def try_prophet_model(time_series):
    """Try to build a Prophet model if the package is available"""
    try:
        from prophet import Prophet
        print("\nBuilding Prophet model...")
        
        # Prepare data for Prophet
        prophet_data = time_series.reset_index()
        prophet_data.columns = ['ds', 'y']
        
        # Create and fit the model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_data)
        
        # Make future dataframe for predictions
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Plot the forecast
        fig = model.plot(forecast)
        plt.title('Prophet Sales Forecast')
        plt.savefig('prophet_forecast.png')
        print("Prophet forecast saved as 'prophet_forecast.png'")
        
        # Plot the components
        fig = model.plot_components(forecast)
        plt.savefig('prophet_components.png')
        print("Prophet components saved as 'prophet_components.png'")
        
        # Return the forecast
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
    
    except ImportError:
        print("\nProphet package not available. Skipping Prophet model.")
        print("To install Prophet, run: pip install prophet")
        return None

def compare_forecasts(time_series, arima_forecast, prophet_forecast=None):
    """Compare different forecasting methods"""
    print("\nComparing forecasting methods...")
    
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(time_series.index, time_series, label='Historical Data')
    
    # Plot ARIMA forecast
    plt.plot(arima_forecast.index, arima_forecast, 'r--', label='ARIMA Forecast')
    
    # Plot Prophet forecast if available
    if prophet_forecast is not None:
        prophet_dates = pd.to_datetime(prophet_forecast['ds'].values)
        plt.plot(prophet_dates, prophet_forecast['yhat'], 'g--', label='Prophet Forecast')
    
    plt.title('Comparison of Forecasting Methods')
    plt.xlabel('Date')
    plt.ylabel('Sales Amount')
    plt.legend()
    plt.grid(True)
    plt.savefig('forecast_comparison.png')
    print("Forecast comparison saved as 'forecast_comparison.png'")

def main():
    print("\n===== TIME SERIES FORECASTING =====\n")
    
    # File path
    file_path = 'sales_data.csv'
    
    # Load data
    daily_sales = load_data(file_path)
    
    # Check stationarity
    is_stationary = check_stationarity(daily_sales['sales_amount'])
    
    # Plot time series components
    plot_time_series_components(daily_sales['sales_amount'])
    
    # Perform seasonal decomposition
    decomposition = seasonal_decomposition(daily_sales['sales_amount'])
    
    # Apply differencing if not stationary
    differenced_series = None
    if not is_stationary:
        differenced_series = difference_series(daily_sales['sales_amount'])
        check_stationarity(differenced_series)
    
    # Determine ARIMA order based on ACF and PACF plots
    # For simplicity, we'll use a standard order, but in practice this should be determined from the plots
    p, d, q = 1, 1, 1
    if is_stationary:
        d = 0
    
    # Build ARIMA model
    if differenced_series is not None and d > 0:
        # If we manually differenced, use d=0 in the ARIMA model
        arima_model = build_arima_model(differenced_series, order=(p, 0, q))
    else:
        arima_model = build_arima_model(daily_sales['sales_amount'], order=(p, d, q))
    
    # Generate ARIMA forecast
    arima_forecast = forecast_with_arima(arima_model)
    
    # Try Prophet model
    prophet_forecast = try_prophet_model(daily_sales['sales_amount'])
    
    # Compare forecasts
    if prophet_forecast is not None:
        compare_forecasts(daily_sales['sales_amount'], arima_forecast, prophet_forecast)
    else:
        compare_forecasts(daily_sales['sales_amount'], arima_forecast)
    
    # Save forecasts to CSV
    arima_forecast_df = pd.DataFrame({
        'date': arima_forecast.index,
        'arima_forecast': arima_forecast.values
    })
    arima_forecast_df.to_csv('arima_forecast.csv', index=False)
    print("\nARIMA forecasts saved to 'arima_forecast.csv'")
    
    if prophet_forecast is not None:
        prophet_forecast.to_csv('prophet_forecast.csv', index=False)
        print("Prophet forecasts saved to 'prophet_forecast.csv'")
    
    print("\n===== TIME SERIES FORECASTING COMPLETE =====\n")

if __name__ == "__main__":
    main()