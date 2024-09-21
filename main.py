#! /usr/bin/env python

#############################################################################
#Name :Kundan ,Erin, Taj, Elliott
#Project : 1

########## imports ###########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import skfuzzy as fuzz
from scipy.spatial.distance import cdist
#################################################################################

class ClusterPlay:
    def __init__(self, file_name):
        # initial class variables
        self.scaled_data = None 
        # loading data
        self.data = pd.read_csv(file_name)
        # preprocessing
        self.data = self.preprocessing(self.data)

    @staticmethod
    def preprocessing(data):
        # checking missing vals
        if data.isna().sum().any(): 
            print("We found missing data.")
        else: 
            print("No missing values.")

        # Dropping customer id 
        data = data.drop("CustomerID", axis=1)
        
        # One-hot encoding for the Gender column 
        data = pd.get_dummies(data, columns=['Gender'], drop_first=True)  # This will create a column 'Gender_Male'

        return data

    def normalize_features(self, features):
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(features)

    def correlation_graph(self):
        corr_matrix = self.data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Graph")
        plt.show()
        plt.savefig("correlation_graph.png")
        plt.close()

    #### added taj code here ########

    def visualize_data(self):
        # Visualize the distribution of age, income, and spending score
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['Annual Income (k$)'], self.data['Spending Score (1-100)'])
        plt.xlabel('Annual Income')
        plt.ylabel('Spending Score')
        plt.title('Customer Data')
        plt.show()
        plt.savefig("customer_data_scatter.png")
        plt.close()

        ######### by gender ############
        plt.figure(figsize=(10, 6))
        for gender in self.data['Gender_Male'].unique():
            plt.scatter(self.data[self.data['Gender_Male'] == gender]['Annual Income (k$)'], 
                        self.data[self.data['Gender_Male'] == gender]['Spending Score (1-100)'], 
                        label='Male' if gender == 1 else 'Female', alpha=0.7)
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.title('Annual Income vs Spending Score by Gender')
        plt.legend()
        plt.show()
        plt.savefig("income_vs_spending_by_gender.png")
        plt.close()


        ########## by age ######################
        age_ranges = [(18, 25), (26, 35), (36, 45), (46, 55), (56, 100)]
        labels = ['18-25', '26-35', '36-45', '46-55', '56+']

        plt.figure(figsize=(10, 6))
        for i, age_range in enumerate(age_ranges):
            plt.scatter(self.data[(self.data['Age'] >= age_range[0]) & (self.data['Age'] <= age_range[1])]['Annual Income (k$)'],self.data[(self.data['Age'] >= age_range[0]) & (self.data['Age'] <= age_range[1])]['Spending Score (1-100)'],label=labels[i], alpha=0.7)

        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.title('Annual Income vs Spending Score by Age Range')
        plt.legend()
        plt.show()
        plt.savefig("income_vs_spending_by_age.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        sns.histplot(self.data['Age'], kde=True)
        plt.title('Age Distribution')

        plt.subplot(1, 3, 2)
        sns.histplot(self.data['Annual Income (k$)'], kde=True)
        plt.title('Income Distribution')

        plt.subplot(1, 3, 3)
        sns.histplot(self.data['Spending Score (1-100)'], kde=True)
        plt.title('Spending Score Distribution')

        plt.tight_layout()
        plt.show()
        plt.savefig("distribution_plots.png")
        plt.close()



    def elbow_method(self):
        inertia = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            inertia.append(kmeans.inertia_)
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, 11), inertia, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.show()
        plt.savefig("elbow.png")
        plt.close()

    def kmeans_clustering(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(self.scaled_data)
        self.data['KMeans Cluster'] = kmeans.labels_
        return kmeans.labels_, kmeans

    ######## added taj code here #################
    def calculate_distortion(self, model):
        # Calculate distortion
        distances = cdist(self.scaled_data, model.cluster_centers_, 'euclidean')
        min_distances = np.min(distances, axis=1)
        distortion = np.mean(min_distances ** 2)
        return distortion

    def fuzzy_cmeans_clustering(self, n_clusters):
        X_scaled_T = self.scaled_data.T
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(X_scaled_T, c=n_clusters, m=2, error=0.005, maxiter=1000)
        fuzzy_labels = np.argmax(u, axis=0)
        self.data['Fuzzy Cluster'] = fuzzy_labels
        return fuzzy_labels, fpc

    def gaussian_mixture_clustering(self, n_clusters):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
        gmm.fit(self.scaled_data)
        gmm_labels = gmm.predict(self.scaled_data)
        self.data['GMM Cluster'] = gmm_labels
        return gmm_labels,gmm

    def evaluate_clustering(self, labels):
        silhouette_avg = silhouette_score(self.scaled_data, labels)
        print(f'Silhouette Score: {silhouette_avg}')
        return silhouette_avg

    def plot_clusters(self, labels, title):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.scaled_data[:, 0], self.scaled_data[:, 1], c=labels, cmap='viridis', s=50)
        plt.title(title)
        plt.xlabel('Feature 1 (scaled)')
        plt.ylabel('Feature 2 (scaled)')
        plt.colorbar(label='Cluster Label')
        plt.show()
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()

def main():
    file_name = "~/project1/Mall_Customers.csv"
    pro = ClusterPlay(file_name)
    # visualize data 
    pro.visualize_data()
    # checking relation between features
    pro.correlation_graph()
    #normalize features
    pro.normalize_features(pro.data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
    # Elbow method
    pro.elbow_method()
    # k-means 
    kmeans_labels, kmeans_model = pro.kmeans_clustering(n_clusters=5)
    pro.evaluate_clustering(kmeans_labels)
    
    # cal distortion
    distortion = pro.calculate_distortion(kmeans_model)
    print(f'Distortion: {distortion}')
    
    # plot k -means cl
    pro.plot_clusters(kmeans_labels, "K-Means Clustering")

    # f k-means
    fuzzy_labels, fpc = pro.fuzzy_cmeans_clustering(n_clusters=5)
    fuzzy_silhouette = pro.evaluate_clustering(fuzzy_labels)
    print(f'Fuzzy Partition Coefficient: {fpc}')

    # plot fkm
    pro.plot_clusters(fuzzy_labels, "Fuzzy C-Means Clustering")

    # gaussian Mixture 
    gmm_labels,gmm_model = pro.gaussian_mixture_clustering(n_clusters=5)
    pro.evaluate_clustering(gmm_labels)
    
    ## for now not able calucuate distortion got to see the methods  is working both fuzz and gaus
    #distortion = pro.calculate_distortion(gmm_model)
    #print(f'Distortion: {distortion}')


    # plot GMM 
    pro.plot_clusters(gmm_labels, "Gaussian Mixture Model Clustering")

if __name__ == "__main__":
    main()
