import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import scrolledtext

def get_data(filename):
    dates = []
    prices = []
    try:
        with open(filename, 'r') as csvfile:
            csvFileReader = csv.reader(csvfile)
            next(csvFileReader)  
            for row in csvFileReader:
                try:
                    date = datetime.strptime(row[0], '%d/%m/%Y')
                    dates.append(date.day)
                    prices.append(float(row[2]))
                except ValueError:
                    print(f"Skipping row due to format issue: {row}")
        if not dates or not prices:
            raise ValueError("No valid data found in the file.")
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None, None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None
    return dates, prices

def train_models(dates, prices):
    dates = np.reshape(dates, (len(dates), 1))
    svr_models = {
        'RBF': SVR(kernel='rbf', C=1e3, gamma=0.1),
        'Linear': SVR(kernel='linear', C=1e3),
        'Polynomial': SVR(kernel='poly', C=1e3, degree=2)
    }

    for name, model in svr_models.items():
        model.fit(dates, prices)
    
    return svr_models

def evaluate_models(models, dates, prices):
    dates = np.reshape(dates, (len(dates), 1))
    errors = {}
    for name, model in models.items():
        predictions = model.predict(dates)
        mse = mean_squared_error(prices, predictions)
        errors[name] = mse
    return errors

def predict_price(models, x):
    x = np.array([[x]])
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(x)[0]
    return predictions

def plot_results(dates, prices, models, x, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(dates, prices, color='black', label='Data')

    for name, model in models.items():
        plt.plot(dates, model.predict(np.reshape(dates, (len(dates), 1))), label=f'{name} model')

    plt.scatter(x, predictions['RBF'], color='red', marker='o', s=100, zorder=5, label=f'Predicted price on day {x}')

    plt.xlabel('Day of the Month')
    plt.ylabel('Price')
    plt.title('Support Vector Regression for Stock Prices')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.show()

def display_output(window, text_widget, message):
    text_widget.insert(tk.END, message + "\n")
    text_widget.see(tk.END)

def create_gui():
    window = tk.Tk()
    window.title("Results")
    
    lbl = tk.Label(window, text="Results")
    lbl.pack(padx=10, pady=10)
    
    text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=20)
    text_area.pack(padx=10, pady=10)
    
    filename = 'nvda.csv'
    dates, prices = get_data(filename)
    
    if dates and prices:
        models = train_models(dates, prices)
        
        errors = evaluate_models(models, dates, prices)
        for name, mse in errors.items():
            display_output(window, text_area, f"{name} model Mean Squared Error: {mse:.2f}")
        
        x_day = 29
        if x_day not in dates:
            display_output(window, text_area, f"Day {x_day} is not in the data range,  model may give inaccurate prediction.")
        predictions = predict_price(models, x_day)

        display_output(window, text_area, f"Predicted price for day {x_day}:")
        for name, pred in predictions.items():
            display_output(window, text_area, f"{name}: {pred:.2f}")

        plot_results(dates, prices, models, x_day, predictions)
    else:
        display_output(window, text_area, "No valid data to process.")

    window.mainloop()

create_gui()
