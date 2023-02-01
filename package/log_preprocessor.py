import pm4py
from pm4py.objects.conversion.log import converter
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization
import pandas as pd




class LogPreprocessor:

    def __init__(self, source_df, case_id_key='hadm_id', activity_key='department', timestamp_key='intime'):
        self.variants = None
        self.source_df = source_df
        self.case_id_key = case_id_key
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key

        self.parameters = {converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: self.case_id_key}
        self.event_log = pm4py.format_dataframe(self.source_df, case_id=self.case_id_key,
                                                activity_key=self.activity_key, timestamp_key=self.timestamp_key)
        self.log = pm4py.convert_to_event_log(self.event_log)
        self.dfg = dfg_discovery.apply(self.log)

    def classify_attributes(proc_c):
        for index, row in proc_c.iterrows():
            if((row["numberOfActivities"] == 1) & (row["numberOfTraceOccurence (Mean)"] == 1)):
                proc_c.at[index, "class"] = "static"
            elif((row["numberOfActivities"] > 1) & (row["numberOfTraceOccurence (Mean)"] == 1)):
                proc_c.at[index, "class"] = "semi-dynamic"
            else:
                proc_c.at[index, "class"] = "dynamic"
        return proc_c    

    def get_process_variants(self):
        # retrieve all possible process variants and remove variants occurring < 20 times due to their small sample size
        self.variants = variants_filter.get_variants(self.log)
        self.variants = list(self.variants.keys())
        var = self.source_df.groupby(self.case_id_key)[self.activity_key].apply(list).reset_index()
        var[self.activity_key] = var[self.activity_key].apply(lambda x: ','.join(map(str, x)))
        var = var.rename({"department":"variant"}, axis=1)
        df_with_variants = self.source_df.merge(var, how="left", on=self.case_id_key)
        var_count = df_with_variants.drop_duplicates(self.case_id_key).groupby('variant').count()
        to_drop = list(var_count.loc[var_count[self.case_id_key] < 20].reset_index()['variant'])
        for ele in to_drop:
            self.variants.remove(ele)
        df_with_variants.rename(columns={self.timestamp_key:"time:timestamp"}, inplace=True)
        df_with_variants["time:timestamp"] = pd.to_datetime(df_with_variants["time:timestamp"])
        return df_with_variants

   