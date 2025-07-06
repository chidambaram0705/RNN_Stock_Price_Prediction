# Stock Price Prediction using RNNs #
This project applies various Recurrent Neural Network (RNN) models—Simple RNN, LSTM, and GRU—to predict stock closing prices for four major companies: Amazon (AMZN), Google (GOOGL), IBM, and Microsoft (MSFT).

**1. Problem Statement**

The goal is to:
- Predict stock closing prices using historical OHLCV (Open, High, Low, Close, Volume) data.
- Explore different RNN architectures.
- Evaluate model performance and identify the most effective approach for time series forecasting.

**2. Dataset**

- Each company's dataset is provided as a CSV file named in the format <company_name>_stock_data.csv.
- The files contain:
Date, Open, High, Low, Close, Volume
- The combined dataset is stored in the variable:
combined_stock_df_cleaned

**3. Workflow Overview**
**A. Data Preprocessing**
- Combined datasets from all 4 companies.
- Removed null values.
- Created rolling windows of data to structure sequences for RNN input.
- Normalized features using MinMaxScaler.

**B. Exploratory Data Analysis (EDA)**
- Correlation analysis between OHLCV.
- Volume distribution and trend visualization.
- Price vs. volume analysis using multi-line subplots.

**C. Model Development**
- Created reusable functions to:
- Generate windowed sequences.
- Scale data.
- Create RNN models using hyperparameter configurations.

**D. Model Training**
- Trained Simple RNN, LSTM, and GRU models.
- Performed hyperparameter tuning using ParameterGrid.

**C. Model Evaluation**
- Evaluated models using metrics like MAE, RMSE, and Loss.
- Visualized actual vs. predicted prices.
- Selected the best configuration based on validation performance.

**4. Best Model Configuration**
- Model Type	Units	Dropout	Learning Rate	Val Loss
GRU	32	0.2	0.0005	0.000002
- This model achieved the best balance of accuracy and generalization.

**5. Final Results**
- The GRU model closely followed the actual stock price trends.
- Prediction plots show the model effectively learns temporal patterns.
- Minor deviations during volatile market periods are observed.

**6. Conclusion**
- GRU outperformed LSTM and Simple RNN in both accuracy and training efficiency.
- Correlation analysis shows a strong linear relationship among OHLC prices.
- Volume is a weaker predictor, but its spikes indicate potential market events.
- Careful window sizing, scaling, and model tuning significantly impact forecasting performance.

## Technologies Used ##
- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
