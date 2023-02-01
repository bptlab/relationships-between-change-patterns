import numpy as np
import matplotlib.pyplot as plt


def plot_hist(df, previous_act, act, measure):
    all_nan = np.all(np.isnan(df.loc[(df['previous_activity'] == previous_act) & (df['department'] == act)][measure].values[0]))
    if not all_nan:
        print(previous_act, act, measure)
        print(df.loc[(df['previous_activity'] == previous_act) & (df['department'] == act)][measure].values[0])
        try:
            plt.figure(figsize=(5,5))
            plt.title(f"{previous_act}->{act}")
            plt.ylabel(f"{measure}")
            plt.hist(df.loc[(df['previous_activity'] == previous_act) & (df['department'] == act)][measure]) 
            plt.show()
        except Exception as e:
            print(e)

def plot_hist_by_row(row, measure='1'):
    fig= plt.figure(figsize=(7,7))
    try:
        plt.title(f"Relation: {row['Act_1']} -> {row['Act_2']} Correlation Coefficient: {row['scipy_corr']}", fontsize=20)
        plt.xlabel(f"{row[f'measure_{measure}']} Distribution", fontsize=18)
        plt.hist(row[f'values_{measure}'])
        plt.show()
    except Exception as e:
        print(e)

def plot_correlation_by_row(row):
    fig= plt.figure(figsize=(7,7))
    try:
        plt.title(f"Relation: {row['Act_1']} -> {row['Act_2']} Correlation Coefficient: {row['scipy_corr']}", fontsize=20)
        plt.xlabel(f"{row['measure_1']}  change", fontsize=18)
        plt.ylabel(f"{row['measure_2']}  change", fontsize=18)
        plt.scatter(row['values_1'], row['values_2'])    
    except Exception as e:
        print(e)
