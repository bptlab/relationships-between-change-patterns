import numpy as np
import pandas as pd
import pm4py
import numpy as np
from pm4py.objects.conversion.log import converter as log_converter
from scipy.stats import variation
from scipy import stats
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.statistics.eventually_follows.log import get as efg_get
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk
from scipy.stats import chi2_contingency
import math
import statistics
import pingouin as pg
import graphviz
from statsmodels.stats import multitest
from statsmodels.stats.contingency_tables import SquareTable as ST
import sys
import sklearn
import scipy
from scipy.stats import shapiro
import tempfile
from copy import copy
from graphviz import Digraph

from pm4py.statistics.attributes.log import get as attr_get
from pm4py.objects.dfg.utils import dfg_utils
from pm4py.util import xes_constants as xes
from pm4py.visualization.common.utils import *
from pm4py.util import exec_utils
from pm4py.statistics.sojourn_time.log import get as soj_time_get
from enum import Enum
from pm4py.util import constants
from typing import Optional, Dict, Any, Tuple
from pm4py.objects.log.obj import EventLog
from pm4py.statistics.start_activities.pandas import get as sa_get
from pm4py.statistics.end_activities.pandas import get as ea_get


from pm4py.visualization.common import save

from matplotlib import pyplot as plt, image as mpimg

from package import visualization
from package import clustering

from package.log_preprocessor import LogPreprocessor
from package import plotting

class ProcessChangeAnalyzer:

    def __init__(self, source_df, activity_identifier, case_id_identifier, time_column, continuous_columns=None, categorical_columns=None):
        self.source_df = source_df
        self.activity_column = activity_identifier
        self.case_id_column = case_id_identifier
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.time_column = time_column

        self.preprocessor = LogPreprocessor(self.source_df, case_id_key=self.case_id_column, activity_key=self.activity_column, timestamp_key=self.time_column)
        self.df_with_variants = self.preprocessor.get_process_variants()

    def prepare_correlation(self):
        self.prepare_continuous_correlation()
        self.prepare_categorical_correlation()
        self.prepare_con_cat_correlation()
    
    def prepare_continuous_correlation(self, first_trace_row_identifier = 'Emergency Department'):
        # for continuous columns
        continuous_diff_df = self.df_with_variants.set_index([self.case_id_column, self.activity_column])[self.continuous_columns].diff().reset_index()
        continuous_diff_df['previous_activity'] = continuous_diff_df[self.activity_column].shift(periods=1)
        continuous_diff_df = continuous_diff_df[continuous_diff_df[self.activity_column] != first_trace_row_identifier]

        # add time perspective
        continuous_diff_df['duration'] = continuous_diff_df['time:timestamp'].dt.seconds
        continuous_diff_df['sum_timestamp'] = continuous_diff_df['duration'].rolling(2).sum()
        continuous_diff_df.drop(['duration', 'time:timestamp'], axis=1, inplace=True)

        self.continuous_diff_df = continuous_diff_df
        continuous_grouped_df = self.continuous_diff_df.groupby([self.activity_column, 'previous_activity']).agg(list)
        self.continuous_grouped_df = continuous_grouped_df
        return self.continuous_grouped_df

    def prepare_categorical_correlation(self):
        diff_df = self.df_with_variants.set_index([self.case_id_column, self.activity_column])[self.categorical_columns].reset_index()
        shifted_df = diff_df.shift(periods = 1)
        joined_df = diff_df.join(shifted_df, rsuffix='_prev')
        joined_df = joined_df[joined_df[self.activity_column] != 'Emergency Department']
        # concat columns
        change_list = []
        for i, row in joined_df.iterrows():
            list_item = {
                self.case_id_column: row[self.case_id_column],
                'previous_activity': row[self.activity_column + '_prev'],
                self.activity_column: row[self.activity_column]
            }
            for col in self.categorical_columns:
                list_item[col] = f"{row[col + '_prev']}-{row[col]}"
            change_list.append(list_item)
        change_df = pd.DataFrame(change_list)
        self.prepared_categorical_df = change_df.groupby(['previous_activity', self.activity_column]).agg(list)
        return self.prepared_categorical_df
    
    def prepare_con_cat_correlation(self):
        con_grouped = self.continuous_grouped_df
        cat_grouped = self.prepared_categorical_df
        con_grouped = con_grouped.reorder_levels(['previous_activity', self.activity_column]).sort_index()
        con_cat = con_grouped.merge(cat_grouped, left_index=True, right_on=["previous_activity", self.activity_column], how="left")
        con_cat.drop(self.case_id_column + '_y', axis=1, inplace=True)
        con_cat.rename(columns={self.case_id_column + '_x':self.case_id_column}, inplace=True)
        self.con_cat = con_cat
        return self.con_cat


    def distribution_checker(self, alpha=0.05):
    # check for distribution of diff values
        distribution_list = []
        for index, temp_row in self.continuous_grouped_df.reset_index().iterrows():
            for i in range(len(temp_row.index)):
                if temp_row.index[i] not in ['previous_activity', 'time:timestamp', self.activity_column, self.case_id_column] and len(temp_row[i]) >= 3:
                    stat, p = shapiro(temp_row[i])
                    distribution_list.append([temp_row['previous_activity'], temp_row[self.activity_column], temp_row.index[i], stat, p, True if p >= alpha else False])

        self.distribution_df = pd.DataFrame(distribution_list, columns=['previous_activity', 'activity', 'measure', 'stat', 'p', 'is_normal_distributed'])
        return self.distribution_df

    def is_measure_normal_distributed(self, previous_act, act, measure):
        return self.distribution_df.loc[(self.distribution_df['previous_activity'] == previous_act) & (self.distribution_df['activity'] == act)  \
            & (self.distribution_df['measure'] == measure)]['is_normal_distributed'].values[0]

    def compute_correlations(self):
        self.compute_correlations_continuous()
        self.compute_correlations_categorical()
        self.compute_correlations_con_cat()
        self._merge_correlation_dfs()

    def compute_correlations_continuous(self):

        # correlate all values for each row
        #i, j columns of row to correlate
        temp_row  = self.continuous_grouped_df.reset_index().iloc[1]
        pearson = []
        spearman = []
        columns_not_to_correlate = ['time:timestamp', 'previous_activity', self.activity_column, self.case_id_column]
        for index, temp_row in self.continuous_grouped_df.reset_index().iterrows():
            for i in range(len(temp_row.index)):
                for j in range(i, len(temp_row.index)):
                    try:
                        if temp_row.index[i] not in columns_not_to_correlate and temp_row.index[j] not in columns_not_to_correlate:
                            # remove measurements w/o values for this transition
                            if ~np.isnan(temp_row[i]).all() and ~np.isnan(temp_row[j]).all():
                                usable_indices = np.intersect1d(np.where(~np.isnan(temp_row[i]))[0], np.where(~np.isnan(temp_row[j]))[0])
                                # scipy.pearsonr only works with 2 or more samples
                                if len(usable_indices) > 2:
                                    # check whether or not both data pairs are normal distributed
                                    if self.is_measure_normal_distributed(temp_row['previous_activity'], temp_row[self.activity_column], temp_row.index[i]) and self.is_measure_normal_distributed(temp_row['previous_activity'], temp_row[self.activity_column], temp_row.index[j]):
                                        scipy_coef = scipy.stats.pearsonr(np.array(temp_row[i])[[usable_indices]], np.array(temp_row[j])[[usable_indices]])
                                        pearson.append([temp_row['previous_activity'], temp_row[self.activity_column], temp_row.index[i], temp_row.index[j], len(usable_indices), scipy_coef[0], np.array(temp_row[i])[[usable_indices]][0], np.array(temp_row[j])[[usable_indices]][0]])        
                                    else:
                                        scipy_coef = scipy.stats.spearmanr(np.array(temp_row[i])[[usable_indices]], np.array(temp_row[j])[[usable_indices]])
                                        spearman.append([temp_row['previous_activity'], temp_row[self.activity_column], temp_row.index[i], temp_row.index[j], len(usable_indices), scipy_coef[0], np.array(temp_row[i])[[usable_indices]][0], np.array(temp_row[j])[[usable_indices]][0]])
                    except Exception as e:
                        print(e)
                                
        self.pearson_df = pd.DataFrame(pearson, columns = ['Act_1', 'Act_2', 'measure_1', 'measure_2', 'sample_size', 'scipy_corr', 'values_1', 'values_2'])
        self.spearman_df= pd.DataFrame(spearman, columns = ['Act_1', 'Act_2', 'measure_1', 'measure_2', 'sample_size', 'scipy_corr', 'values_1', 'values_2'])
        
    def compute_correlations_con_cat(self):
        con_columns = self.continuous_columns.copy()
        con_columns.remove("time:timestamp")
        con_columns.append("sum_timestamp")
        cat_columns = self.categorical_columns.copy()
        con_cat = self.con_cat
        anova = []
        kruskal = []
        for index, temp_row in con_cat.reset_index().iterrows():
            for con in con_columns:
                for cat in cat_columns:
                    try:
                    # remove measurements w/o values for this transition
                        if ~np.isnan(temp_row[con]).all() and ~(np.array(temp_row[cat]) == 'nan-nan').all():
                            usable_indices = np.intersect1d(
                                np.where(~np.isnan(temp_row[con]))[0],                             
                                np.flatnonzero(np.core.defchararray.find(temp_row[cat],'nan')<0)
                            )
                            if len(usable_indices) >= 2:
                                cat_dict = {}
                                cat_usable = np.array(temp_row[cat])[[usable_indices]]
                                unique_cats = np.unique(cat_usable)
                                #get values for each category
                                for unique_cat in unique_cats:
                                    cat_index = np.where(np.array(temp_row[cat]) == unique_cat)[0]
                                    cat_usable_index = np.intersect1d(cat_index, usable_indices)
                                    con_unique_cat = np.array(temp_row[con])[[cat_usable_index]] 
                                    cat_dict[unique_cat] = con_unique_cat
                                if len(cat_dict) > 1:
                                    if self.is_measure_normal_distributed(temp_row['previous_activity'], temp_row[self.activity_column], con):
                                        #perform ANOVA one way test
                                        anova_test = stats.f_oneway(*(cat_dict[v] for v in cat_dict))
                                        anova.append([temp_row['previous_activity'], temp_row[self.activity_column], con, cat, len(usable_indices), anova_test[0], anova_test[1]])
                                    else:
                                        #perform kruskal-wallis test        
                                        kruskal_test = stats.kruskal(*(cat_dict[v] for v in cat_dict))
                                        kruskal.append([temp_row['previous_activity'], temp_row[self.activity_column], con, cat, len(usable_indices), kruskal_test[0], kruskal_test[1]])
                    except Exception as e:
                            print(e)
        self.anova_df = pd.DataFrame(anova, columns = ['Act_1', 'Act_2', 'measure_1', 'measure_2', 'sample_size', 'stat', 'p'])
        self.kruskal_df = pd.DataFrame(kruskal, columns = ['Act_1', 'Act_2', 'measure_1', 'measure_2', 'sample_size', 'stat', 'p'])
        return self.anova_df, self.kruskal_df
    
    def compute_correlation_of_one_with_all_cells(self, act_1, act_2, measure, var_1):
        con_columns = self.continuous_columns.copy()
        con_columns.remove("time:timestamp")
        con_columns.append("sum_timestamp")
        cat_columns = self.categorical_columns.copy()

        con_cat = self.con_cat
        corr_arr = []
        for index, temp_row in con_cat.reset_index().iterrows():
            act_3 = temp_row['previous_activity']
            act_4 = temp_row[self.activity_column]
            if act_3 == act_1 and act_4 == act_2:
                pass
            else:
                for col in con_cat.columns:
                    if col != "hadm_id":
                        corr, p, s, sample_size, method = self.compute_correlation_for_single_cell(act_1, act_2, act_3, act_4, measure, col, "", "")
                        corr_arr.append([act_1, act_2, act_3, act_4, measure, col, sample_size, corr, p, s, method])
        corr_df = pd.DataFrame(corr_arr, columns=['Act_1', 'Act_2', 'Act_3', 'Act_4',  'measure_1', 'measure_2', 'sample_size', 'scipy_corr', 'p', 'stat', 'method'])
        corr_df = corr_df.loc[corr_df["sample_size"] > 0].reset_index().drop("index", axis=1)
        return corr_df
                

    
    def compute_correlation_for_single_cell(self, act_1, act_2, act_3, act_4, eA_1, eA_2, var_1, var_2):
        
        con_columns = self.continuous_columns.copy()
        con_columns.remove("time:timestamp")
        con_columns.append("sum_timestamp")
        cat_columns = self.categorical_columns.copy()
        con_cat = self.con_cat
        
        #Get data types of event attributes
        first_ea_type = ""
        second_ea_type = ""
        
        if eA_1 in con_columns:
            first_ea_type = "con"
        else:
            first_ea_type = "cat"


        if eA_2 in con_columns:
            second_ea_type = "con"
        else:
            second_ea_type = "cat"
        
        row_rel_1 = con_cat.loc[(act_1, act_2)]
        row_rel_2 = con_cat.loc[(act_3, act_4)]
        usable_case_ids = np.intersect1d(row_rel_1[self.case_id_column], row_rel_2[self.case_id_column])
        usable_case_ids_1 = np.isin(row_rel_1[self.case_id_column], usable_case_ids)
        usable_case_ids_2 = np.isin(row_rel_2[self.case_id_column], usable_case_ids)
        usable_case_index_1 = np.where(usable_case_ids_1 == True)[0]
        usable_case_index_2 = np.where(usable_case_ids_2 == True)[0]
        usable_values_1 = np.array(row_rel_1[eA_1])[[usable_case_index_1]]
        usable_values_2 = np.array(row_rel_2[eA_2])[[usable_case_index_2]]
        #two functions - one for usable indices, one for statistical tests and cat preprocessing
        usable_indices = self.retrieve_usable_indices(first_ea_type, second_ea_type, usable_values_1, usable_values_2)
        if len(usable_indices) <= 2:
            print("Intersection of usable values is less than 2. Cannot calculate correlation")
            return (0, 1, 0, 0, "")
        else:
            corr, p, s, method = self.calculate_correlation(first_ea_type, second_ea_type, usable_values_1, usable_values_2, usable_indices, act_1, act_2, act_3, act_4, eA_1, eA_2)
        return corr, p, s, len(usable_indices), method

    def calculate_correlation(self, first_ea_type, second_ea_type, usable_values_1, usable_values_2, usable_indices, act_1, act_2, act_3, act_4, eA_1, eA_2):
        p = 1
        s = 0
        corr = 0
        method = ""

        if first_ea_type == "con" and second_ea_type == "con":
            if ~np.isnan(usable_values_1).all() and ~np.isnan(usable_values_2).all():
                if self.is_measure_normal_distributed(act_1, act_2, eA_1) and self.is_measure_normal_distributed(act_3, act_4, eA_2):
                    scipy_coef = scipy.stats.pearsonr(np.array(usable_values_1)[[usable_indices]], np.array(usable_values_2)[[usable_indices]])
                    corr = scipy_coef[0]
                    method = 'pearson'
                else:
                    scipy_coef = scipy.stats.spearmanr(np.array(usable_values_1)[[usable_indices]], np.array(usable_values_2)[[usable_indices]])
                    corr = scipy_coef[0]
                    method = 'spearman'
        elif first_ea_type == "con" and second_ea_type == "cat":
            if ~np.isnan(usable_values_1).all() and ~(np.array(usable_values_2) == 'nan-nan').all():
                cat_dict = {}
                cat_usable = np.array(usable_values_2)[[usable_indices]]
                unique_cats = np.unique(cat_usable)
                #get values for each category
                for unique_cat in unique_cats:
                    cat_index = np.where(np.array(usable_values_2) == unique_cat)[0]
                    cat_usable_index = np.intersect1d(cat_index, usable_indices)
                    con_unique_cat = np.array(usable_values_1)[[cat_usable_index]] 
                    cat_dict[unique_cat] = con_unique_cat
                if len(cat_dict) > 1:
                    if self.is_measure_normal_distributed(act_1, act_2, eA_1):
                        #perform ANOVA one way test
                        anova_test = stats.f_oneway(*(cat_dict[v] for v in cat_dict))
                        p = anova_test[1]
                        s = anova_test[0]
                        method = 'anova'
                    else:
                        #perform kruskal-wallis test        
                        kruskal_test = stats.kruskal(*(cat_dict[v] for v in cat_dict))
                        p = kruskal_test[1]
                        s = kruskal_test[0]
                        method = 'kruskal'
        elif first_ea_type == "cat" and second_ea_type == "con":
            if ~np.isnan(usable_values_2).all() and ~(np.array(usable_values_1) == 'nan-nan').all():
                cat_dict = {}
                cat_usable = np.array(usable_values_1)[[usable_indices]]
                unique_cats = np.unique(cat_usable)
                #get values for each category
                for unique_cat in unique_cats:
                    cat_index = np.where(np.array(usable_values_1) == unique_cat)[0]
                    cat_usable_index = np.intersect1d(cat_index, usable_indices)
                    con_unique_cat = np.array(usable_values_2)[[cat_usable_index]] 
                    cat_dict[unique_cat] = con_unique_cat
                if len(cat_dict) > 1:
                    if self.is_measure_normal_distributed(act_3, act_4, eA_2):
                        #perform ANOVA one way test
                        anova_test = stats.f_oneway(*(cat_dict[v] for v in cat_dict))
                        p = anova_test[1]
                        s = anova_test[0]
                        method = 'anova'
                    else:
                        #perform kruskal-wallis test        
                        kruskal_test = stats.kruskal(*(cat_dict[v] for v in cat_dict))
                        p = kruskal_test[1]
                        s = kruskal_test[0]
                        method = 'kruskal'
        elif first_ea_type == "cat" and second_ea_type == "cat":
            if ~(np.array(usable_values_1) == 'nan-nan').all() and ~(np.array(usable_values_2) == 'nan-nan').all():
                corr = self.cramer_v(np.array(usable_values_1)[[usable_indices]], np.array(usable_values_2)[[usable_indices]])
                method = 'cramer'
        return corr, p, s, method
        

        


    def retrieve_usable_indices(self, first_ea_type, second_ea_type, usable_values_1, usable_values_2):
        usable_indices = []
        if first_ea_type == "con" and second_ea_type == "con":
            if ~np.isnan(usable_values_1).all() and ~np.isnan(usable_values_2).all(): 
                usable_indices = np.intersect1d(np.where(~np.isnan(usable_values_1))[0], np.where(~np.isnan(usable_values_2))[0])
        elif first_ea_type == "con" and second_ea_type == "cat":
            if ~np.isnan(usable_values_1).all() and ~(np.array(usable_values_2) == 'nan-nan').all():
                usable_indices = np.intersect1d(
                    np.where(~np.isnan(usable_values_1))[0],                             
                    np.flatnonzero(np.core.defchararray.find(usable_values_2,'nan')<0))
        elif first_ea_type == "cat" and second_ea_type == "con":
            if ~np.isnan(usable_values_2).all() and ~(np.array(usable_values_1) == 'nan-nan').all():
                usable_indices = np.intersect1d(
                    np.where(~np.isnan(usable_values_2))[0],                             
                    np.flatnonzero(np.core.defchararray.find(usable_values_1,'nan')<0))
        elif first_ea_type == "cat" and second_ea_type == "cat":
            if ~(np.array(usable_values_1) == 'nan-nan').all() and ~(np.array(usable_values_2) == 'nan-nan').all():
                usable_indices = np.intersect1d(
                    np.flatnonzero(np.core.defchararray.find(usable_values_1,'nan')<0),                             
                    np.flatnonzero(np.core.defchararray.find(usable_values_2,'nan')<0))
        return usable_indices



    def _merge_correlation_dfs(self):
        ## merge pearson and spearson df
        self.pearson_df = self.pearson_df.dropna()
        self.spearman_df = self.spearman_df.dropna()
        self.cramer_df  = self.cramer_df.dropna()
        self.pearson_df['method'] = 'pearson'
        self.spearman_df['method'] = 'spearman'
        self.cramer_df['method'] = 'cramer'
        self.anova_df['method'] = 'anova'
        self.kruskal_df['method'] = 'kruskal'
        self.correlation_df = pd.concat([self.pearson_df, self.spearman_df, self.cramer_df, self.anova_df, self.kruskal_df])
        self.correlation_df = self.correlation_df[["Act_1", "Act_2", "measure_1", "measure_2", "sample_size", "scipy_corr", "p", "stat", "method"]]
        self.continuous_correlation_df = pd.concat([self.pearson_df, self.spearman_df])
    


    
    def cramer_v(self, values_1, values_2):
        # generate contingency table
        (val_1, val_2), count = stats.contingency.crosstab(values_1, values_2)
        # Finding Chi-squared test statistic, 
        # sample size, and minimum of rows and
        # columns
        X2 = stats.chi2_contingency(count, correction=False)[0]
        N = np.sum(count)
        minimum_dimension = min(count.shape)-1

        # Calculate Cramer's V
        return(np.sqrt((X2/N) / minimum_dimension))

    def get_contingency_table(self, previous_activity, activity, measurement_1, measurement_2):
        temp_row = self.get_correlation_row_by_measurment( previous_activity, activity, measurement_1, measurement_2)
        (val_1, val_2), count = stats.contingency.crosstab(temp_row['values_1'],temp_row['values_2'])
        print(f"{temp_row['Act_1']} -> {temp_row['Act_2']} for {measurement_1} to {measurement_2} with {temp_row['scipy_corr']}")
        print('x', val_2)
        print('y', val_1)
        print(count)

    def compute_correlations_categorical(self):
        # correlate all values for each row
        craemers_v = []
        for index, temp_row in self.prepared_categorical_df.reset_index().iterrows():
            for i in range(len(temp_row.index)):
                for j in range(i+1, len(temp_row.index)):
                    try:
                        if temp_row.index[i] not in ['previous_activity', self.activity_column, self.case_id_column] and temp_row.index[j] not in ['previous_activity', self.activity_column, self.case_id_column]:
                            # remove measurements w/o values for this transition
                            if ~(np.array(temp_row[i]) == 'nan-nan').all() and ~(np.array(temp_row[j]) == 'nan-nan').all():
                                usable_indices = np.intersect1d(
                                    np.flatnonzero(np.core.defchararray.find(temp_row[i],'nan')<0),                             
                                    np.flatnonzero(np.core.defchararray.find(temp_row[j],'nan')<0)
                                )
                                if len(usable_indices) >= 2:
                                    coef = self.cramer_v(np.array(temp_row[i])[[usable_indices]], np.array(temp_row[j])[[usable_indices]])
                                    craemers_v.append([temp_row['previous_activity'], temp_row[self.activity_column], temp_row.index[i], temp_row.index[j], len(usable_indices), coef, np.array(temp_row[i])[[usable_indices]][0], np.array(temp_row[j])[[usable_indices]][0]])
                    except Exception as e:
                        #print(f"{temp_row['department_prev']} -> {temp_row['department']} for {temp_row.index[i]} to {temp_row.index[j]} with {coef[0][1]}")
                        print(e)
        self.cramer_df = pd.DataFrame(craemers_v, columns = ['Act_1', 'Act_2', 'measure_1', 'measure_2', 'sample_size', 'scipy_corr', 'values_1', 'values_2'])
        return self.cramer_df

    def filter_correlation_by_measurements(self, measurements):
        return self.correlation_df[self.correlation_df['measure_1'].isin(measurements) & self.correlation_df['measure_2'].isin(measurements) & (self.correlation_df['measure_1'] != self.correlation_df['measure_2'])]
    

    def enrich_event_log(self, enrichment_method='none'):
        pass

    def get_correlation_row_by_measurment(self, previous_activity, activity, measurement_1, measurement_2):
        return self.correlation_df[ \
            (self.correlation_df['Act_1'] == previous_activity) & \
            (self.correlation_df['Act_2'] == activity) & \
            (self.correlation_df['measure_1'] == measurement_1) & \
            (self.correlation_df['measure_2'] == measurement_2) \
            ].iloc[0] 

    def filter_strong_correlations(self,df=None):
        if df is None:
            df = self.correlation_df
        significant_correlations = df[((df['scipy_corr'] > 0.6) | (df['scipy_corr'] < -0.6)) & (df['measure_1'] != df['measure_2']) & (df['sample_size'] > 10)]
        significant_correlations['abs_corr'] = np.abs(significant_correlations['scipy_corr'])
        significant_correlations = significant_correlations.sort_values('abs_corr', ascending=False)
        return significant_correlations

    def show_correlation_frequency_by_measurement_pair(self, df):
        return pd.DataFrame(self.filter_strong_correlations(df).groupby(['measure_1', 'measure_2']).size().sort_values(ascending=False))

    def get_process_characteristics_correlation(self):
        return self.continuous_correlation_df[self.continuous_correlation_df['measure_2'] == 'sum_timestamp']

    def visualize_row(self, previous_activity, activity, measurement_1, measurement_2):
        row = self.get_correlation_row_by_measurment(previous_activity, activity, measurement_1, measurement_2)
        plotting.plot_correlation_by_row(row)
        plotting.plot_hist_by_row(row, '1')
        plotting.plot_hist_by_row(row, '2')

    def cluster(self, clustering_attributes, n_clusters):
        c = clustering.Clustering(self.continuous_diff_df, clustering_attributes, n_clusters)
        clustering_result = c.cluster()
        clustering_result = clustering_result.drop(clustering_attributes, axis=1)
        clustered_df = copy(self.source_df)
        self.clustered_df = clustered_df.set_index('hadm_id').join(clustering_result)
        return self.clustered_df

    def edge_correlation_coefficient(self, edge, allowed_attributes):

        res = self.correlation_df.loc[
                (self.correlation_df['Act_1'] == edge[0]) &  \
                (self.correlation_df['Act_2'] == edge[1]) & \
                (self.correlation_df['measure_1'].isin(allowed_attributes)) & \
                (self.correlation_df['measure_2'].isin(allowed_attributes)), \
                 "scipy_corr"]
        res = res.sort_values(ascending=False)
        if len(res.values) > 0:
            return res.values[0]
        else:
            return 0



    def visualize(self, attributes):
        edges_correlation = copy(self.preprocessor.dfg)
        for edge in edges_correlation:
            edges_correlation[edge] = self.edge_correlation_coefficient(edge, attributes)

        start_activities = sa_get.get_start_activities(self.df_with_variants)
        end_activities = ea_get.get_end_activities(self.df_with_variants)


        gviz = visualization.apply(self.preprocessor.dfg, edges_correlation=edges_correlation,
                                   parameters={"start_activities": start_activities, "end_activities": end_activities})

        fig = plt.figure(figsize=(20,20))

        file_name = tempfile.NamedTemporaryFile(suffix='.png')
        file_name.close()

        save.save(gviz, file_name.name)

        img = mpimg.imread(file_name.name)
        plt.axis('off')
        plt.imshow(img)
        plt.show()

