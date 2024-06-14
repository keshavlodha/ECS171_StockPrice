from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

lr_model = joblib.load("LR_model.pkl")
dt_model = joblib.load("DT_model.pkl")
rf_model = joblib.load("RF_model.pkl")
models = {'lr': lr_model, "dt": dt_model, "rf":rf_model}

def plot_last_month(symbol):
    # Load stock data
    sp500_stocks = pd.read_csv('sp500_stocks.csv')
    sp500_stocks['Date'] = pd.to_datetime(sp500_stocks['Date'])
    
    # Filter the data for the given symbol and last month
    last_month = sp500_stocks[sp500_stocks['Symbol'] == symbol].tail(30)
    
    plt.figure(figsize=(10, 5))
    plt.plot(last_month['Date'], last_month['Close'], label='Close Price')
    plt.title(f'Last Month Stock Prices for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join('static', f'{symbol}_last_month.png')    
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        symbol = request.form.get("symbol")
        date = request.form.get("date")
        model_type = request.form.get("model_type")

        if models is None or model_type not in models:
            return "Model is not loaded. Please check the server logs for more details."

        model = models[model_type]

        # Preprocess the input data for your model (if needed)
        sp500_stocks = pd.read_csv('sp500_stocks.csv')
        sp500_stocks['Date'] = pd.to_datetime(sp500_stocks['Date'])

        sp500_stocks['Prev_Close'] = sp500_stocks.groupby('Symbol')['Close'].shift(1)
        sp500_stocks['MA_5'] = sp500_stocks.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5).mean())
        sp500_stocks['MA_10'] = sp500_stocks.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=10).mean())
        sp500_stocks['MA_20'] = sp500_stocks.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).mean())
        sp500_stocks.fillna(method='bfill', inplace=True)

        company_data = sp500_stocks[sp500_stocks['Symbol'] == symbol].sort_values('Date')
        if company_data.empty:
            return f"No data available for symbol: {symbol}"

        latest_data = company_data.iloc[-1]

        X_new = pd.DataFrame({
            'Prev_Close': [latest_data['Prev_Close']],
            'MA_5': [latest_data['MA_5']],
            'MA_10': [latest_data['MA_10']],
            'MA_20': [latest_data['MA_20']]
        })

        # Make prediction using your model
        prediction = model.predict(X_new)
        # Format the prediction for display
        predicted_price = prediction[0]  # Assuming single class output

        plot_path = plot_last_month(symbol)


        return render_template("result.html", prediction=predicted_price, plot_path=plot_path)

    else:
        return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)
