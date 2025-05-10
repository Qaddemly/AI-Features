# Job Recommendation System Evaluation
  This project evaluates six job recommendation system models using four different clustering techniques. The goal is to assess the performance of each model in terms of cluster purity and execution time.

## Models
#### the following models are evaluated 
      1. SBERT (Sentence-BERT)

      2. TF-IDF (Term Frequency-Inverse Document Frequency)

      3. LDA (Latent Dirichlet Allocation)

      4. KNN (K-Nearest Neighbors)

      5. BM25 (Best Matching 25)

      6. FastText

## Clustering Techniques 
  Each model is evaluated using the following clustering techniques:

### 1) K-means Clustering
    Number of Clusters: 6

    Description: K-means is a centroid-based clustering algorithm that partitions the data into k clusters by minimizing the variance within each cluster.

    Usage: Used to evaluate the cluster purity of job recommendations.

### 2) DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    Parameters:

    eps=0.5 (maximum distance between two samples to be considered neighbors)

    min_samples=5 (minimum number of points to form a cluster)

    Description: DBSCAN groups points based on their density in the feature space and can handle noise and outliers.

    Usage: Evaluates how well the models perform in identifying dense regions of job embeddings.

### 3)Spectral Clustering
    Number of Clusters: 20

    Description: Spectral Clustering uses the eigenvalues of a similarity matrix to perform dimensionality reduction before clustering in fewer dimensions.

    Usage: Evaluates the models' ability to capture complex structures in the data.

### 4) BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
    Number of Clusters: 19

    Description: BIRCH is a hierarchical clustering algorithm that is efficient for large datasets. It builds a tree-like structure to summarize the data before clustering.

    Usage: Evaluates the models' performance on large-scale job recommendation tasks.

## Evaluation Metrics
  The primary evaluation metric is Cluster Purity, which measures the percentage of recommended jobs that belong to the same cluster as the resume. The formula for cluster purity is:

  Cluster Purity=  (Number of recommended jobs in the same cluster as the resume) / (Total number of recommended jobs)



## Results
  The results for each model and clustering technique are summarized below:

  Model     	K-means   	DBSCAN      	Spectral Clustering     	BIRCH         AVG  

  SBERT	      75.00%	    65.00%	          90.00%              	65.00%        73.00%
  TF-IDF	    100.00%    	100.00%	          100.00%	              100.00%       100.00%
  LDA	        100.00%	    0.00%          	  95.00%	              0.00%         48.00%
  KNN	        100.00%	    100.00%	          100.00%	              100.00%       100.00%
  BM25	      100.00%	    100.00%	          100.00%	              100.00%       100.00%
  FastText	  0.00%	      0.00%	            0.00%              	  0.00%         0.00%



