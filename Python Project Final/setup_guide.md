# Setup Guide for Sales Prediction Model

## Installation Instructions

1. **Install Required Packages**

   Before running the sales prediction model, you need to install the required Python packages. Open a command prompt or PowerShell window in the project directory and run:

   ```
   pip install -r requirements.txt
   ```

   This will install all the necessary dependencies listed in the requirements.txt file:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn

2. **Run the Model**

   After installing the required packages, you can run the sales prediction model with:

   ```
   python sales_prediction_model.py
   ```

3. **Review Results**

   The script will generate several visualization files in the project directory:
   - sales_analysis.png
   - sales_analysis_additional.png
   - correlation_matrix.png
   - model_performance.png
   - feature_importance.png
   - sales_forecast.png

   It will also create a CSV file with future sales predictions:
   - sales_predictions.csv

## Troubleshooting

- If you encounter a "No module named" error, make sure you've installed all the required packages using the command in step 1.
- If you're using a virtual environment, ensure it's activated before installing packages and running the script.
- For visualization issues, ensure you have a backend that supports matplotlib (this is usually automatic on most systems).