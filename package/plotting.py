import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

def plot_cell_correlation(row):
    
    if row["method"] == "pearson" or row["method"] == "spearman":
        fig, ax = plt.subplots(figsize=(10,4))
        plt.title(f"Relation: {row['Act_1']} -> {row['Act_2']} Correlation Coefficient: {row['scipy_corr'].round(2)}", fontsize=10)
        plt.xlabel(f"{row['measure_1']}  change", fontsize=10)
        plt.ylabel(f"{row['measure_2']}  change", fontsize=10)
        plt.scatter(row['values_1'], row['values_2'])   
    elif row["method"] == "cramer":
        ct = pd.crosstab(index=row['values_1'],columns=row['values_2']).plot(kind="bar", figsize=(10,4))
        plt.title(f"Relation: {row['Act_1']} -> {row['Act_2']} Correlation Coefficient: {row['scipy_corr'].round(2)}", fontsize=10)
        plt.xlabel(f"{row['measure_1']}  change", fontsize=10)
        plt.ylabel(f"{row['measure_2']}  change", fontsize=10)
        ax = plt.gca()
        plt.xticks(size=5, wrap=True)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    elif row["method"] == "anova" or row["method"] == "kruskal":
        fig, ax = plt.subplots(figsize=(10,4))
        sns.swarmplot(x='values_1', y='values_2', data=row, dodge=True, palette='viridis')
        plt.title(f"Relation: {row['Act_1']} -> {row['Act_2']} Correlation Coefficient: {row['stat'].round(2)}", fontsize=10)
        plt.xlabel(f"{row['measure_2']}  change", fontsize=10)
        plt.ylabel(f"{row['measure_1']}  change", fontsize=10)  
        ax = plt.gca()
        plt.xticks(size=5, wrap=True)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')   
        #draw mean(avg) line
        sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="values_1",
            y="values_2",
            data=row,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax)

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
