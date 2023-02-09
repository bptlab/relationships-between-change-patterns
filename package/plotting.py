import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_cell_correlation(row):
    figsize_value = (4,3.75)
    fontsize_value = 5
    if row["method"] == "pearson" or row["method"] == "spearman":
        plt.close(2)
        fig, ax = plt.subplots(figsize=figsize_value, num='Correlation Plot')
        plt.title(f"Relation: {row['Act_1']} -> {row['Act_2']} Correlation Coefficient: {row['scipy_corr'].round(2)}", fontsize=fontsize_value)
        plt.xlabel(f"{row['measure_1']}  change", fontsize=fontsize_value)
        plt.ylabel(f"{row['measure_2']}  change", fontsize=fontsize_value)
        plt.xticks(size=5, wrap=True)
        plt.yticks(size=5)
        plt.scatter(row['values_1'], row['values_2'])   
    elif row["method"] == "cramer":
        plt.close(2)
        pd.crosstab(index=row['values_1'],columns=row['values_2']).plot(kind="bar", figsize=figsize_value, num='Correlation Plot')
        legend = plt.legend(title=f"{row['measure_2']}  change", prop={'size': 6}, title_fontsize=6)
        l_texts = [item for item in legend.get_texts()]
        for l_text in l_texts:
            l_text.set_text(l_text.get_text().replace('abnormal ', ''))
        plt.title(f"Relation: {row['Act_1']} -> {row['Act_2']} Correlation Coefficient: {row['scipy_corr'].round(2)}", fontsize=fontsize_value)
        plt.xlabel(f"{row['measure_1']}  change", fontsize=fontsize_value)
        plt.ylabel(f"{row['measure_2']}  change", fontsize=fontsize_value)
        ax = plt.gca()
        plt.xticks(size=5, wrap=True)
        plt.yticks(size=5)
        plt.setp(ax.get_xticklabels(), rotation=7.5, horizontalalignment='center')
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels_new = [label.replace('abnormal ', '') for label in labels]
        ax.set_xticklabels(labels_new)
    elif row["method"] == "anova" or row["method"] == "kruskal":
        plt.close(2)
        fig, ax = plt.subplots(figsize=figsize_value, num='Correlation Plot')
        sns.swarmplot(x='values_1', y='values_2', data=row, dodge=True, palette='viridis')
        plt.title(f"Relation: {row['Act_1']} -> {row['Act_2']} Correlation Coefficient: {row['stat'].round(2)}", fontsize=fontsize_value)
        plt.xlabel(f"{row['measure_2']}  change", fontsize=fontsize_value)
        plt.ylabel(f"{row['measure_1']}  change", fontsize=fontsize_value)  
        plt.xticks(size=5, wrap=True)
        plt.yticks(size=5)  
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
        plt.setp(ax.get_xticklabels(), rotation=7.5, horizontalalignment='center')
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels_new = [label.replace('abnormal ', '') for label in labels]
        ax.set_xticklabels(labels_new)
