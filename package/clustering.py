from sklearn.cluster import KMeans
import pandas as pd

class Clustering:

    def __init__(self, df, attributes, n_clusters):
        self.df = df
        self.n_clusters = n_clusters
        self.attributes = attributes
        self.additional_attributes = ['hadm_id'] 
        self.prepare_df()

    def prepare_df(self):
        self.clustering_df = self.df[self.additional_attributes + self.attributes]
        self.clustering_df = self.clustering_df.groupby('hadm_id').min()
        self.clustering_df = self.clustering_df.dropna()


    def cluster(self):
        model = KMeans(n_clusters=self.n_clusters)
        model.fit(self.clustering_df)
        all_predictions = model.predict(self.clustering_df)
        self.clustering_df['cluster'] = all_predictions

        return self.clustering_df