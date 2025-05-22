# Sales Prediction Model - User Guide

## Overview

This project provides two complementary approaches to sales prediction:

1. **Machine Learning Approach** (`sales_prediction_model.py`): Uses Random Forest and Linear Regression to predict sales based on various features including product category, region, and promotion status.

2. **Time Series Forecasting** (`time_series_forecasting.py`): Uses statistical time series methods (ARIMA) and Facebook's Prophet to capture seasonal patterns and trends in the sales data.

## Installation

Before using the models, install all required dependencies:

```
pip install -r requirements.txt
```

Note: The Prophet package may require additional system dependencies. If you encounter issues, refer to the [Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html).

## Using the Models

### Machine Learning Model

Run the machine learning model with:

```
python sales_prediction_model.py
```

This will:
- Load and preprocess the sales data
- Perform exploratory data analysis with visualizations
- Build and evaluate Linear Regression and Random Forest models
- Generate future sales predictions
- Save all visualizations and predictions to files

### Time Series Forecasting

Run the time series forecasting with:

```
python time_series_forecasting.py
```

This will:
- Analyze the stationarity of the time series
- Decompose the series into trend, seasonal, and residual components
- Build an ARIMA model for forecasting
- Attempt to build a Prophet model (if installed)
- Compare different forecasting methods
- Save all visualizations and forecasts to files

## Interpreting the Results

### Visualizations

1. **sales_analysis.png** and **sales_analysis_additional.png**:
   - Shows sales trends over time, by category, by region, and the effect of promotions
   - Helps identify patterns and relationships in the historical data

2. **correlation_matrix.png**:
   - Shows how different variables relate to each other
   - Strong positive correlations appear in red, negative in blue

3. **model_performance.png**:
   - Compares actual vs. predicted sales for both models
   - Points closer to the diagonal line indicate better predictions

4. **feature_importance.png**:
   - Shows which features have the most influence on sales predictions
   - Higher importance means the feature has more predictive power

5. **sales_forecast.png**:
   - Shows historical sales and future predictions from the machine learning model

6. **time_series_diagnostics.png**:
   - Shows autocorrelation and partial autocorrelation functions
   - Helps identify appropriate parameters for the ARIMA model

7. **seasonal_decomposition.png**:
   - Breaks down the time series into trend, seasonal, and residual components
   - Helps understand underlying patterns in the data

8. **arima_forecast.png** and **prophet_forecast.png**:
   - Show forecasts from the respective time series models

9. **forecast_comparison.png**:
   - Compares forecasts from different models
   - Helps identify which model might be more appropriate for your data

### Output Files

1. **sales_predictions.csv**:
   - Contains date and predicted sales from the machine learning model

2. **arima_forecast.csv**:
   - Contains date and predicted sales from the ARIMA model

3. **prophet_forecast.csv** (if Prophet is installed):
   - Contains date, predicted sales, and confidence intervals from the Prophet model

## Choosing Between Models

- **Machine Learning Model**: Better when you have many features that influence sales (promotions, product categories, etc.) and want to understand their impact.

- **Time Series Forecasting**: Better for capturing seasonal patterns and trends when you have a longer history of data and are primarily interested in the time component.

- For the most robust predictions, consider an ensemble approach that combines both methods.

## Customizing the Models

### Machine Learning Model

To modify the machine learning model, edit `sales_prediction_model.py`:
- Change the features used for prediction
- Try different machine learning algorithms
- Adjust the train/test split ratio

### Time Series Forecasting

To modify the time series forecasting, edit `time_series_forecasting.py`:
- Change the ARIMA order parameters (p, d, q)
- Adjust the forecasting horizon
- Modify the seasonal decomposition period

## Updating with New Data

To update the models with new sales data:

1. Add new records to `sales_data.csv` following the same format
2. Run the models again to generate updated predictions

## Troubleshooting

- **Missing module errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Visualization errors**: Make sure you have a working matplotlib backend
- **Prophet installation issues**: Follow the specific installation instructions for your platform from the Prophet documentation