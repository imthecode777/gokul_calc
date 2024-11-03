from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Simulate random data for industrial operations
np.random.seed(0)
n_samples = 100

# Generate random input data
labor_hours = np.random.uniform(40, 60, n_samples)
machine_downtime = np.random.uniform(0, 10, n_samples)
material_quality = np.random.uniform(0.5, 1.0, n_samples)
maintenance_frequency = np.random.uniform(0, 5, n_samples)
energy_costs = np.random.uniform(10, 20, n_samples)
production_volume = np.random.uniform(1000, 5000, n_samples)
error_rate = np.random.uniform(0, 0.2, n_samples)

# Generate outputs based on a theoretical relationship
productivity = 0.5 * labor_hours - 0.3 * machine_downtime + 0.7 * material_quality + np.random.normal(0, 1, n_samples)
cost = 1.2 * labor_hours + 0.5 * energy_costs + 0.3 * maintenance_frequency + 0.8 * machine_downtime + np.random.normal(0, 1, n_samples)
profit = 2 * productivity - 1.5 * cost + np.random.normal(0, 1, n_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Labor Hours': labor_hours,
    'Machine Downtime': machine_downtime,
    'Material Quality': material_quality,
    'Maintenance Frequency': maintenance_frequency,
    'Energy Costs': energy_costs,
    'Production Volume': production_volume,
    'Error Rate': error_rate,
    'Productivity': productivity,
    'Cost': cost,
    'Profit': profit
})

# Define the model and features for each target variable
features = ['Labor Hours', 'Machine Downtime', 'Material Quality', 'Maintenance Frequency', 'Energy Costs']

# Model productivity
X = data[features]
y_productivity = data['Productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y_productivity, test_size=0.2, random_state=42)
model_productivity = LinearRegression().fit(X_train, y_train)

# Model cost
y_cost = data['Cost']
X_train, X_test, y_train, y_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
model_cost = LinearRegression().fit(X_train, y_train)

# Model profit (uses Productivity and Cost as inputs)
y_profit = data['Profit']
X_profit = data[['Productivity', 'Cost']]
X_train, X_test, y_train, y_test = train_test_split(X_profit, y_profit, test_size=0.2, random_state=42)
model_profit = LinearRegression().fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array([[
        data['labor_hours'],
        data['machine_downtime'],
        data['material_quality'],
        data['maintenance_frequency'],
        data['energy_costs']
    ]])
    
    # Make predictions
    productivity_pred = model_productivity.predict(input_data)[0]
    cost_pred = model_cost.predict(input_data)[0]
    profit_pred = model_profit.predict([[productivity_pred, cost_pred]])[0]
    
    results = {
        "productivity": productivity_pred,
        "cost": cost_pred,
        "profit": profit_pred
    }
    
    return jsonify(results)

@app.route('/plots')
def plots():
    plot_paths = generate_plots()
    return jsonify(plot_paths)

def generate_plots():
    # Directory to save plots
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    
    plot_paths = []
    
    # Labor Hours vs Productivity
    plt.figure()
    plt.scatter(data['Labor Hours'], data['Productivity'], color='blue', alpha=0.5)
    plt.title('Labor Hours vs Productivity')
    plt.xlabel('Labor Hours')
    plt.ylabel('Productivity')
    labor_vs_productivity_path = 'static/plots/labor_vs_productivity.png'
    plt.savefig(labor_vs_productivity_path)
    plot_paths.append(labor_vs_productivity_path)
    plt.close()
    
    # Machine Downtime vs Productivity
    plt.figure()
    plt.scatter(data['Machine Downtime'], data['Productivity'], color='orange', alpha=0.5)
    plt.title('Machine Downtime vs Productivity')
    plt.xlabel('Machine Downtime')
    plt.ylabel('Productivity')
    downtime_vs_productivity_path = 'static/plots/downtime_vs_productivity.png'
    plt.savefig(downtime_vs_productivity_path)
    plot_paths.append(downtime_vs_productivity_path)
    plt.close()
    
    # Energy Costs vs Cost
    plt.figure()
    plt.scatter(data['Energy Costs'], data['Cost'], color='green', alpha=0.5)
    plt.title('Energy Costs vs Cost')
    plt.xlabel('Energy Costs')
    plt.ylabel('Cost')
    energy_vs_cost_path = 'static/plots/energy_vs_cost.png'
    plt.savefig(energy_vs_cost_path)
    plot_paths.append(energy_vs_cost_path)
    plt.close()
    
    # Productivity vs Profit
    plt.figure()
    plt.scatter(data['Productivity'], data['Profit'], color='purple', alpha=0.5)
    plt.title('Productivity vs Profit')
    plt.xlabel('Productivity')
    plt.ylabel('Profit')
    productivity_vs_profit_path = 'static/plots/productivity_vs_profit.png'
    plt.savefig(productivity_vs_profit_path)
    plot_paths.append(productivity_vs_profit_path)
    plt.close()
    
    # Cost vs Profit
    plt.figure()
    plt.scatter(data['Cost'], data['Profit'], color='red', alpha=0.5)
    plt.title('Cost vs Profit')
    plt.xlabel('Cost')
    plt.ylabel('Profit')
    cost_vs_profit_path = 'static/plots/cost_vs_profit.png'
    plt.savefig(cost_vs_profit_path)
    plot_paths.append(cost_vs_profit_path)
    plt.close()
    
    return plot_paths

if __name__ == '__main__':
    app.run(debug=True)
