# Sales Prediction Model

This project implements a machine learning model to predict future sales based on historical sales data. It includes data preprocessing, exploratory data analysis with visualizations, and sales forecasting.

## Project Structure

- `sales_data.csv`: Historical sales data including date, product information, region, sales amount, units sold, and promotion status
- `sales_prediction_model.py`: Main Python script that performs data analysis and builds the prediction model
- `sales_predictions.csv`: Output file containing future sales predictions

## Visualizations Generated

The model generates several visualizations to help understand the sales data:

1. `sales_analysis.png`: Basic sales analysis including time series, category breakdown, regional analysis, and promotion effects
2. `sales_analysis_additional.png`: Additional insights including sales distribution, units vs amount correlation, monthly trends, and day-of-week patterns
3. `correlation_matrix.png`: Correlation between different numeric features
4. `model_performance.png`: Comparison of actual vs predicted sales for both Linear Regression and Random Forest models
5. `feature_importance.png`: Top features influencing the sales prediction model
6. `sales_forecast.png`: Visualization of historical sales and future predictions

## How to Run

1. Ensure you have Python installed with the required packages:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn

2. Run the prediction model:
   ```
   python sales_prediction_model.py
   ```

3. Review the generated visualizations and the `sales_predictions.csv` file for forecasted sales.

## Model Details

The project implements two regression models:
1. Linear Regression - A simple baseline model
2. Random Forest Regressor - A more complex ensemble model that typically provides better predictions for this type of data

The models are evaluated using Root Mean Squared Error (RMSE) and R² score metrics.

## Features Used for Prediction

- Time-based features (year, month, day, day of week)
- Product category (one-hot encoded)
- Region (one-hot encoded)
- Promotion status
- Time index (for capturing trends)

The feature importance visualization shows which factors most strongly influence sales predictions.