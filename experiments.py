import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)
num_average_time = 10  # Number of times to run each experiment to calculate the average values

# Function to create fake data (take inspiration from usage.py)
def generate_data(N, M, input_type, output_type):
  if(input_type == 'real'):
    X = pd.DataFrame(np.random.randn(N, M))
  else:
    X = pd.DataFrame({i: pd.Series(np.random.randint(M, size=N), dtype="category") for i in range(M)})

  if(output_type == 'real'):
    y = pd.Series(np.random.randn(N))
  else:
    y = pd.Series(np.random.randint(M, size=N), dtype="category")
  return X, y

# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def calculate_time(N, M, input_type, output_type, test_size, criterias, num_average_time):
  X, y = generate_data(N, M, input_type, output_type)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
  time_data = {}
  # Stores data for each criteria

  for criteria in criterias:
    fit_times = []
    prediction_times = []
    for i in range(num_average_time):
      tree = DecisionTree(criteria, max_depth=5)
      start_fit_time = time.time()
      tree.fit(X_train, y_train)
      end_fit_time = time.time()
      fit_times.append(end_fit_time-start_fit_time)

      start_prediction_time = time.time()
      tree.predict(X_test)
      end_prediction_time = time.time()
      prediction_times.append(end_prediction_time-start_prediction_time)
    
    avg_fit_time = np.mean(fit_times)
    avg_prediction_time = np.mean(prediction_times)

    std_fit_time = np.std(fit_times)
    std_prediction_time = np.std(prediction_times)

    print(f"For criteria {criteria}, average train time is {avg_fit_time} and average test time is {avg_prediction_time}")
    print(f"For criteria {criteria}, standard deviation for training time is {std_fit_time} and standard deviation for testing data is {std_prediction_time}")

    time_data[criteria] = {
      "train_time" : avg_fit_time,
      "test_time" : avg_prediction_time,
      "std_train_time" : std_fit_time,
      "std_test_time" : std_prediction_time
    }
  return time_data

# ...
# Function to plot the results
'''
Results are indexed as following
[input_type][output_type] [criteria] -> {dictionary: time_data that we have formed by calculate_time function}
Plot will be of x-axis = N and y-axis = Time taken
'''

def plot_results(results, N_values, M_values, criteria):
  save_dir = "./Q1.4_plot"
  os.makedirs(save_dir, exist_ok=True)
  # First varrying N_values keeping M constant

  print("\nTime vs number of samples(N)\n")
  plt.figure(figsize=(16, 5*len(M_values)))

  for i, M in enumerate(M_values):
    ax1 = plt.subplot(len(M_values), 2, 2*i+1)
    ax2 = plt.subplot(len(M_values), 2, 2*i+2)
    for input_type in ['discrete', 'real']:
      for output_type in ['discrete', 'real']:
        train_times = []
        prediction_times = []

        for N in N_values:
          data = results[(N, M)][(input_type, output_type)][criteria] # This will be dictionary
          train_times.append(data['train_time'])
          prediction_times.append(data['test_time'])
        ax1.plot(N_values, train_times, marker='o', label = f"{input_type} - {output_type}")
        ax2.plot(N_values, prediction_times, marker = 'o', label = f"{input_type} - {output_type}")

    ax1.set_xlabel("Number of Samples (N)")
    ax1.set_ylabel("Training Time (seconds)")
    ax1.set_title(f"Training Time vs Number of Samples (N), M = {M}")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel("Number of Samples (N)")
    ax2.set_ylabel("Prediction Time (seconds)")
    ax2.set_title(f"Prediction Time vs Number of Samples (N), M = {M}")
    ax2.legend()
    ax2.grid(True)

  plt.savefig(f"{save_dir}/time_vs_N_{criteria}.png", bbox_inches='tight', dpi=300)
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.2, hspace=0.25)
  plt.show()

  # Now varrying M_values keeping N constant
  print("\nTime vs Number of Features (M)\n")
  plt.figure(figsize=(16, 5*len(M_values)))

  for i, N in enumerate(N_values):
    ax1 = plt.subplot(len(N_values), 2, 2*i+1)
    ax2 = plt.subplot(len(N_values), 2, 2*i+2)
    for input_type in ['discrete', 'real']:
      for output_type in ['discrete', 'real']:
        train_times = []
        prediction_times = []

        for M in M_values:
          data = results[(N, M)][(input_type, output_type)][criteria] # This will be dictionary
          train_times.append(data['train_time'])
          prediction_times.append(data['test_time'])
        ax1.plot(M_values, train_times, marker='o', label = f"{input_type} - {output_type}")
        ax2.plot(M_values, prediction_times, marker = 'o', label = f"{input_type} - {output_type}")

    ax1.set_xlabel("Number of Features (M)")
    ax1.set_ylabel("Training Time (seconds)")
    ax1.set_title(f"Training Time vs Number of Features (M), N = {N}")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel("Number of Features (M)")
    ax2.set_ylabel("Prediction Time (seconds)")
    ax2.set_title(f"Prediction Time vs Number of Features (M), N = {N}")
    ax2.legend()
    ax2.grid(True)

  plt.savefig(f"{save_dir}/time_vs_M_{criteria}.png", bbox_inches='tight', dpi=300)
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.2, hspace=0.25)
  plt.show()


# ...
# Other functions

def find_results(N_values, M_values, input_types, output_types, test_size, criterias, num_average_time):
  results = {}
  for N in N_values:
    for M in M_values:
      results[(N, M)] = {}
      for input_type in input_types:
        for output_type in output_types:
          data_i = calculate_time(N, M, input_type, output_type, test_size, criterias, num_average_time)
          results[(N, M)][(input_type, output_type)] = data_i
  return results

# ...
# Run the functions, Learn the DTs and Show the results/plots

N_values = [10, 20, 50]
M_values = [1, 5, 10]
criterias = ['information_gain', 'gini_index']
input_types = ['real', 'discrete']
output_types = ['real', 'discrete']
test_size = 0.3

results = find_results(N_values, M_values, input_types, output_types, test_size, criterias, num_average_time)

plot_results(results, N_values, M_values, criteria='information_gain')
plot_results(results, N_values, M_values, criteria='gini_index')