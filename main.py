#! /usr/bin/env python

#############################################################################
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
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
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
        
        # One-hot encoding for the Genre column 
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

    # replacing seprate elbow method and distortion is included in it with graph
    def elbow_method_kmeans(self, max_clusters=10):
        inertia = []
        distortion = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(self.scaled_data)
            inertia.append(kmeans.inertia_)
            if i > 1:  # Distortion is only calculated for 2 or more clusters
                distances = cdist(self.scaled_data, kmeans.cluster_centers_, 'euclidean')
                min_distances = np.min(distances, axis=1)
                distortion.append(np.mean(min_distances ** 2))
            else:
                distortion.append(None)  # No distortion for 1 cluster

        # Plotting Elbow Method and Distortion
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(range(1, max_clusters + 1), inertia, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for K-Means')

        plt.subplot(2, 1, 2)
        plt.plot(range(2, max_clusters + 1), distortion[1:], marker='o')  # Start from 2 clusters
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distortion')
        plt.title('Distortion for K-Means')

        plt.tight_layout()
        plt.show()
        plt.savefig("elbow_&_distortion_kmeans.png")
        plt.close()

    def elbow_method_fuzzy(self, max_clusters=10):
        fpc_fcm = []
        distortion = []
        for i in range(1, max_clusters + 1):
            cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(self.scaled_data.T, c=i, m=2, error=0.005, maxiter=1000)
            fpc_fcm.append(fpc)
            if i > 1:  # Distortion is only calculated for 2 or more clusters
                distances = cdist(self.scaled_data, cntr, 'euclidean')
                min_distances = np.min(distances, axis=1)
                distortion.append(np.mean(min_distances ** 2))
            else:
                distortion.append(None)  # No distortion for 1 cluster

        # Plotting Elbow Method and Distortion
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(range(1, max_clusters + 1), fpc_fcm, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Fuzzy Partition Coefficient')
        plt.title('Elbow Method for Fuzzy C-Means')

        plt.subplot(2, 1, 2)
        plt.plot(range(2, max_clusters + 1), distortion[1:], marker='o')  # Start from 2 clusters
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distortion')
        plt.title('Distortion for Fuzzy C-Means')

        plt.tight_layout()
        plt.show()
        plt.savefig("elbow_fuzzy.png")
        plt.close()

    ### not using this #########
    def elbow_method(self):
        inertia = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(self.scaled_data)
            inertia.append(kmeans.inertia_)
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, 11), inertia, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.savefig("elbow.png")
        plt.close()

    def kmeans_clustering(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(self.scaled_data)
        self.data['KMeans Cluster'] = kmeans.labels_
        return kmeans.labels_, kmeans

    ########### not using this ##################
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
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(self.scaled_data)
        gmm_labels = gmm.predict(self.scaled_data)
        self.data['GMM Cluster'] = gmm_labels
        return gmm_labels,gmm

    def evaluate_clustering(self, labels):
        silhouette_avg = silhouette_score(self.scaled_data, labels)
        print(f'Silhouette Score: {silhouette_avg}')
        return silhouette_avg

    def plot_clusters(self, labels, title, algo_kmean=False,gmm_model=None):
        # Reduce dimensions using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.scaled_data)
        plt.figure(figsize=(10, 8))
        # Create a scatter plot for the clusters
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
        # Add a legend for the clusters
        unique_labels = np.unique(labels)
        for label in unique_labels:
            plt.scatter([], [], color=scatter.cmap(scatter.norm(label)), label=f'Cluster {label}')
        if algo_kmean:
            # Calculate centroids in the PCA space
            centroids_pca = []
            for label in unique_labels:
                cluster_data = self.scaled_data[labels == label]
                centroid = np.mean(cluster_data, axis=0)
                centroids_pca.append(pca.transform([centroid]))
            # Add centroids to the plot (only one entry in the legend)
            centroid_color = 'yellow'
            for centroid in centroids_pca:
                plt.scatter(centroid[0][0], centroid[0][1], s=100, c=centroid_color, edgecolor='black', marker='o')

            # Add a single legend entry for centroids
            plt.scatter([], [], s=100, color=centroid_color, edgecolor='black', label='Centroids')
            # Draw ellipses for GMM clusters
        if gmm_model is not None:
            for i in range(gmm_model.n_components):
                # Get the mean and covariance of the cluster
                mean = gmm_model.means_[i]
                cov = gmm_model.covariances_[i]
                # Calculate the eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                # Calculate the angle of the ellipse
                angle = np.arctan2(eigenvectors[0][1], eigenvectors[0][0])
                angle = np.degrees(angle)
                # Calculate the width and height of the ellipse
                width, height = 2 * np.sqrt(eigenvalues)
                # Create the ellipse
                ellipse = plt.matplotlib.patches.Ellipse(mean, width, height, angle=angle, color='red', alpha=0.5)
                plt.gca().add_patch(ellipse)

        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.show()
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()
    def plot_clusters(self, labels, title, algo_kmean=False, gmm_model=None):
        # Reduce dimensions using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.scaled_data)
        plt.figure(figsize=(10, 8))
        # Create a scatter plot for the clusters
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
        # Add a legend for the clusters
        unique_labels = np.unique(labels)
        for label in unique_labels:
            plt.scatter([], [], color=scatter.cmap(scatter.norm(label)), label=f'Cluster {label}')
        if algo_kmean:
            # Calculate centroids in the PCA space
            centroids_pca = []
            for label in unique_labels:
                cluster_data = self.scaled_data[labels == label]
                centroid = np.mean(cluster_data, axis=0)
                centroids_pca.append(pca.transform([centroid]))
            # Add centroids to the plot (only one entry in the legend)
            for centroid in centroids_pca:
                plt.scatter(centroid[0][0], centroid[0][1], s=100, c='yellow', edgecolor='black', marker='o')
        # Add a single legend entry for centroids
        plt.scatter([], [], s=100, color="yellow", edgecolor='black', label='Centroids')
        # Draw ellipses for GMM clusters
        if gmm_model is not None:
            for i in range(gmm_model.n_components):
                # Get the mean and covariance of the cluster
                mean = gmm_model.means_[i]
                cov = gmm_model.covariances_[i]
                # Calculate the eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                # Ensure we only take the first two eigenvalues and eigenvectors for 2D plotting
                if len(eigenvalues) > 2:
                    eigenvalues = eigenvalues[:2]
                    eigenvectors = eigenvectors[:, :2]
                    # Calculate the angle of the ellipse
                angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                angle = np.degrees(angle)
            
                # Calculate the width and height of the ellipse
                width, height = 2 * np.sqrt(np.abs(eigenvalues))
            
                # Create the ellipse
                ellipse = plt.matplotlib.patches.Ellipse(mean, width, height, angle=angle, color='red', alpha=0.5)
                plt.gca().add_patch(ellipse)

        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.show()
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()

    def aic_method_gaussian(self, max_clusters=10):
        aic_values = []
        for i in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=i, random_state=24)
            gmm.fit(self.scaled_data)
            aic_values.append(gmm.aic(self.scaled_data))

        # Plotting AIC values
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), aic_values, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('AIC')
        plt.title('AIC for Gaussian Mixture Model')
        plt.xticks(range(1, max_clusters + 1))
        plt.grid()
        plt.savefig("aic_gaussian_mixture.png")
        plt.show()
        plt.close()
    
    #fuc override
    def evaluate_clustering(self, labels):
        silhouette_avg = silhouette_score(self.scaled_data, labels)
        dbi = davies_bouldin_score(self.scaled_data, labels)
        ch_index = calinski_harabasz_score(self.scaled_data, labels)
        
        print(f'Silhouette Score: {silhouette_avg}')
        print(f'Davies-Bouldin Index: {dbi}')
        print(f'Calinski-Harabasz Index: {ch_index}')
        


def main():
    file_name = "/jet/home/suddapal/project1_1/Mall_Customers.xls"
    pro = ClusterPlay(file_name)
    # visualize data 
    pro.visualize_data()
    # checking relation between features
    pro.correlation_graph()
    #normalize features
    #print(pro.data.head())
    pro.normalize_features(pro.data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
    # elbow method
    # pro.elbow_method()
    ################# k-means #################################################

    #"""
    ## elbow method include both distortion ,elbow and graph
    print("K means")
    pro.elbow_method_kmeans()
    kmeans_labels, kmeans_model = pro.kmeans_clustering(n_clusters=4)
    pro.evaluate_clustering(kmeans_labels)
    # plot k -means cl
    #print(kmeans_labels)
    pro.plot_clusters(kmeans_labels, "K-Means Clustering",True)
    #### neeed to summary clutter analysis form taj


    ################# k-means ##################################################

    
    ########################## fuzzy  k-means######################################
    print("fuzzy k means")
    pro.elbow_method_fuzzy()
    fuzzy_labels, fpc = pro.fuzzy_cmeans_clustering(n_clusters=4)
    fuzzy_silhouette = pro.evaluate_clustering(fuzzy_labels)
    print(f'Fuzzy Partition Coefficient: {fpc}')
    # plot fkm
    pro.plot_clusters(fuzzy_labels, "Fuzzy C-Means Clustering")
    #### neeed to summary clutter analysis form taj
    ######################### fuzzy k-means ########################################
    #"""

    ######################## gaussian Mixture ####################################
    print("gaussian")

    pro.aic_method_gaussian()
    gmm_labels,gmm_model = pro.gaussian_mixture_clustering(n_clusters=5)
    pro.evaluate_clustering(gmm_labels)

    # plot GMM 
    pro.plot_clusters(gmm_labels, "Gaussian Mixture Model Clustering",False,gmm_model)
    ####################### gaussian Mixture #####################################

if __name__ == "__main__":
    main()

