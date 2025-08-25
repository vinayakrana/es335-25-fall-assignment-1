import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)

def generate_data(N, M, input_type, output_type):
    if input_type == "real":
        X = pd.DataFrame(np.random.randn(N, M))
    else:  # discrete
        X = pd.DataFrame({i: pd.Series(np.random.randint(M, size=N), dtype="category") for i in range(M)})

    if output_type == "real":
        y = pd.Series(np.random.randn(N))
    else:  # discrete
        y = pd.Series(np.random.randint(M, size=N), dtype="category")

    return X, y 



def evaluate_runtime(N, M, input_type, output_type, test_size, criterias, num_average_time):
    X, y = generate_data(N, M, input_type, output_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    time_data = {}

    for criteria in criterias:
        tree = DecisionTree(criterion=criteria)
        # Measure fit time
        fit_times = []
        
        for _ in range(num_average_time):
            start_time = time.perf_counter()
            tree.fit(X_train, y_train)
            end_time = time.perf_counter()
            fit_times.append(end_time - start_time)
        avg_fit_time = np.mean(fit_times)
        std_fit_time = np.std(fit_times)

        # Measure predict time
        predict_times = []
        for _ in range(num_average_time):
            start_time = time.perf_counter()
            tree.predict(X_test)
            end_time = time.perf_counter()
            predict_times.append(end_time - start_time)
        avg_predict_time = np.mean(predict_times)
        std_predict_time = np.std(predict_times)

        print(f"Criteria: {criteria}")
        print(f"Average fit time: {avg_fit_time:.4f} ± {std_fit_time:.4f}")
        print(f"Average predict time: {avg_predict_time:.4f} ± {std_predict_time:.4f}")
        time_data[criteria] = {
            
            'fit_time': (avg_fit_time, std_fit_time),
            'predict_time': (avg_predict_time, std_predict_time)
        }


def plot_combined_time_complexity_N(df, criterion='information_gain'):
    unique_M = np.sort(df['M'].unique())
    for M in unique_M:
        single_M_df = df[(df['M'] == M) & (df['criterion'] == criterion)]
       

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
        axes = axes.ravel()

        for (in_type, out_type), group in single_M_df.groupby(['input_type', 'output_type']):
            grp = group.sort_values('N')
            label = f"{in_type}-{out_type}"
            axes[0].plot(grp['N'], grp['train_time'], marker='o', label=label)
            axes[1].plot(grp['N'], grp['test_time'], marker='o', label=label)

        axes[0].set_title(f'Training time vs N (M={M}, criterion={criterion})')
        axes[0].set_xlabel('N (number of samples)')
        axes[0].set_ylabel('Training time (s)')
        axes[0].grid(True)
        axes[0].legend()

        axes[1].set_title(f'Prediction time vs N (M={M}, criterion={criterion})')
        axes[1].set_xlabel('N (number of samples)')
        axes[1].set_ylabel('Prediction time (s)')
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

def plot_combined_time_complexity_M(df, criterion='information_gain'):
    unique_N = np.sort(df['N'].unique())
    for N in unique_N:
        single_N_df = df[(df['N'] == N) & (df['criterion'] == criterion)]
      

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
        axes = axes.ravel()

        for (in_type, out_type), group in single_N_df.groupby(['input_type', 'output_type']):
            grp = group.sort_values('M')
            label = f"{in_type}-{out_type}"
            axes[0].plot(grp['M'], grp['train_time'], marker='o', label=label)
            axes[1].plot(grp['M'], grp['test_time'], marker='o', label=label)

        axes[0].set_title(f'Training time vs M (N={N}, criterion={criterion})')
        axes[0].set_xlabel('M (number of features)')
        axes[0].set_ylabel('Training time (s)')
        axes[0].grid(True)
        axes[0].legend()

        axes[1].set_title(f'Prediction time vs M (N={N}, criterion={criterion})')
        axes[1].set_xlabel('M (number of features)')
        axes[1].set_ylabel('Prediction time (s)')
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()