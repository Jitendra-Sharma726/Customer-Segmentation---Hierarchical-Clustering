import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage


class RetailClustering:

    def __init__(self, filepath):

        print("Loading Online Retail dataset... please wait")

        self.df = pd.read_csv(filepath)

        print(f"Dataset loaded with {len(self.df)} rows")


    def prepare_data(self):

        print("Preparing dataset...")

        data = self.df.select_dtypes(include=['int64', 'float64'])

        data = data.dropna()

        return data


    def perform_clustering(self, data):

        print("Performing hierarchical clustering...")

        linkage_matrix = linkage(data, method='ward')

        return linkage_matrix


    def generate_dendrogram(self, linkage_matrix):

        print("Generating dendrogram... please wait")

        plt.figure(figsize=(10,6))

        dendrogram(linkage_matrix)

        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Data Points")
        plt.ylabel("Distance")

        plt.savefig("dendrogram.png")

        plt.close()

        print("Dendrogram saved successfully")


def run_clustering(filepath):

    model = RetailClustering(filepath)

    data = model.prepare_data()

    linkage_matrix = model.perform_clustering(data)

    model.generate_dendrogram(linkage_matrix)

    return True


if __name__ == "__main__":

    run_clustering("online_retail.csv")

    print("\nHierarchical clustering completed.")
