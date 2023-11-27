import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
# Replace 'customer_data.csv' with the path to your dataset
data = pd.read_csv('Mall_Customers.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Assuming the dataset has columns representing purchase behavior
# Select the relevant columns for clustering
# Adjust these column names to match the columns in your dataset
selected_columns = ['Annual Income (k$)','Spending Score (1-100)']

# Select data from these columns
X = data[selected_columns]

# Perform standardization to normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Determine the number of clusters (K)
k = 4  # You can experiment with different values for K

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_normalized)
# Get the cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_
# Visualize the clusters (for 2D data)
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='red', label='Centroids')
plt.title('K-means Clustering of Customer Purchase History')
plt.xlabel('Annual_Income ')
plt.ylabel('Spending_Score')
plt.legend()
plt.show()
# Assign the cluster labels to your original dataset
data['Cluster'] = cluster_labels
