## K-Means

Source: [http://www.naftaliharris.com/blog/visualizing-k-means-clustering/](http://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

[https://www.datascience.com/blog/k-means-clustering](https://www.datascience.com/blog/k-means-clustering)

A cluster refers to a collection of data points aggregated together because of certain similarities.

K-Means clustering intends to partition n objects into k clusters in which each object belongs to the cluster with the nearest mean. This method produces exactly k different clusters of greatest possible distinction. The best number of clusters k leading to the greatest separation \(distance\) is not known as a priori and must be computed from the data. The objective of K-Means clustering is to minimize total intra-cluster variance, or, the squared error function:

![](/assets/kmeans.png)

You’ll define a target number _k_, which refers to the number of centroids you need in the dataset. A centroid is the imaginary or real location representing the center of the cluster. Every data point is allocated to each of the clusters through reducing the in-cluster sum of squares.

In other words, the K-means algorithm identifies _k \_number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible. The_‘means’ \_in the K-means refers to averaging of the data; that is, finding the centroid.

#### **How the K-means algorithm works**

**Algorithm        **

1. Clusters the data into k groups where k  is predefined.
2. Select k points at random as cluster centers
3. Assign objects to their closest cluster center according to the Euclidean distance function.
4. Calculate the centroid or mean of all objects in each cluster.
5. Repeat steps 2, 3 and 4 until the same points are assigned to each cluster in consecutive rounds.

![](/assets/k-means1.png)

It halts creating and optimizing clusters when either:

* The centroids have stabilized — there is no change in their values because the clustering has been successful.
* The defined number of iterations has been achieved.

K-Means is relatively an efficient method. However, we need to specify the number of clusters, in advance and the final results are sensitive to initialization and often terminates at a local optimum. Unfortunately there is no global theoretical method to find the optimal number of clusters. A practical approach is to compare the outcomes of multiple runs with different k and choose the best one based on a predefined criterion. In general, a large k probably decreases the error but increases the risk of overfitting.

### Choosing K

The algorithm described above finds the clusters and data set labels for a particular pre-chosen_K_. To find the number of clusters in the data, the user needs to run the _K_-means clustering algorithm for a range of_K_ values and compare the results. In general, there is no method for determining exact value of _K_, but an accurate estimate can be obtained using the following techniques.

One of the metrics that is commonly used to compare results across different values of _K_ is the mean distance between data points and their cluster centroid. Since increasing the number of clusters will always reduce the distance to data points, increasing _K_ will_always\_decrease this metric, to the extreme of reaching zero when \_K_ is the same as the number of data points. Thus, this metric cannot be used as the sole target. Instead, mean distance to the centroid as a function of _K_ is plotted and the "elbow point," where the rate of decrease sharply shifts, can be used to roughly determine _K_.

A number of other techniques exist for validating _K_, including cross-validation, information criteria, the information theoretic jump method, the silhouette method, and the G-means algorithm. In addition, monitoring the distribution of data points across groups provides insight into how the algorithm is splitting the data for each _K_.

![](https://www.datascience.com/hs-fs/hubfs/Blog/introduction-to-k-means-clustering-elbow-point-example.png?width=760&height=411&name=introduction-to-k-means-clustering-elbow-point-example.png "introduction-to-k-means-clustering-elbow-point-example.png")

## Business Uses

The_K_-means clustering algorithm is used to find groups which have not been explicitly labeled in the data. This can be used to confirm business assumptions about what types of groups exist or to identify unknown groups in complex data sets. Once the algorithm has been run and the groups are defined, any new data can be easily assigned to the correct group.

This is a versatile algorithm that can be used for any type of grouping. Some examples of use cases are:

* Behavioral segmentation:
  * Segment by purchase history
  * Segment by activities on application, website, or platform
  * Define personas based on interests
  * Create profiles based on activity monitoring
* Inventory categorization:
  * Group inventory by sales activity
  * Group inventory by manufacturing metrics
* Sorting sensor measurements:
  * Detect activity types in motion sensors
  * Group images
  * Separate audio
  * Identify groups in health monitoring
* Detecting bots or anomalies:
  * Separate valid activity groups from bots
  * Group valid activity to clean up outlier detection

In addition, monitoring if a tracked data point switches between groups over time can be used to detect meaningful changes in the data.

