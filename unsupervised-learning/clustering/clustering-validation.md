## Clustering Validation

The term **cluster validation **is used to design the procedure of evaluating the goodness of clustering algorithm results. This is important to avoid finding patterns in a random data, as well as, in the situation where you want to compare two clustering algorithms. Clustering validation considers 2 metrics: Compactness on how close the points are within cluster and Separability, how distinctly separable points are in different clusters.

Generally, clustering validation statistics can be categorized into 3 classes\(Charrad et al. 2014,Brock et al. \(2008\),Theodoridis and Koutroumbas \(2008\)\):

1. **Internal cluster validation**, which uses the internal information of the clustering process to evaluate the goodness of a clustering structure without reference to external information. It can be also used for estimating the number of clusters and the appropriate clustering algorithm without any external data.
2. **External cluster validation**, which consists in comparing the results of a cluster analysis to an externally known result, such as externally provided class labels. It measures the extent to which cluster labels match externally supplied class labels. Since we know the “true” cluster number in advance, this approach is mainly used for selecting the right clustering algorithm for a specific data set.
3. **Relative cluster validation**, which evaluates the clustering structure by varying different parameter values for the same algorithm \(e.g.,: varying the number of clusters k\). It’s generally used for determining the optimal number of clusters.

![](/assets/Screenshot 2019-08-10 at 5.20.45 PM.png)![](/assets/Screenshot 2019-08-10 at 5.23.09 PM.png)![](/assets/Screenshot 2019-08-10 at 5.24.46 PM.png)![](/assets/Screenshot 2019-08-10 at 5.27.43 PM.png)![](/assets/Screenshot 2019-08-10 at 6.36.16 PM.png)![](/assets/Screenshot 2019-08-10 at 6.37.21 PM.png)![](/assets/Screenshot 2019-08-10 at 6.38.43 PM.png)

Silhoutte coefficient should not be used for DBSCCAN as the score rewards compact, dense well separated clusters but does not take into coonsideration noise.![](/assets/Screenshot 2019-08-10 at 6.39.40 PM.png)

Silhoutte coefficient shortcomings![](/assets/Screenshot 2019-08-10 at 6.42.27 PM.png)

Validation for DBSCAN: Density Based Clustering Validation - http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=83C3BD5E078B1444CB26E243975507E1?doi=10.1.1.707.9034&rep=rep1&type=pdf

