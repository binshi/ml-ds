## DBSCAN\(Density based spatial clustering with noise\)

![](/assets/dbscan.png)

![](/assets/Screenshot 2019-08-10 at 12.31.25 PM.png)

[Visualizing DBSCAN Clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

Based on a set of points \(let’s think in a bidimensional space as exemplified in the figure\), DBSCAN groups together points that are close to each other based on a distance measurement \(usually Euclidean distance\) and a minimum number of points. It also marks as outliers the points that are in low-density regions.

#### Parameters:

The DBSCAN algorithm basically requires 2 parameters:

> **eps**: specifies how close points should be to each other to be considered a part of a cluster. It means that if the distance between two points is lower or equal to this value \(eps\), these points are considered neighbors.
>
> **minPoints**: the minimum number of points to form a dense region. For example, if we set the minPoints parameter as 5, then we need at least 5 points to form a dense region.

#### Parameter estimation:

The parameter estimation is a problem for every data mining task. To choose good parameters we need to understand how they are used and have at least a basic previous knowledge about the data set that will be used.

> **eps**: if the eps value chosen is too small, a large part of the data will not be clustered. It will be considered outliers because don’t satisfy the number of points to create a dense region. On the other hand, if the value that was chosen is too high, clusters will merge and the majority of objects will be in the same cluster. The eps should be chosen based on the distance of the dataset \(we can use a k-distance graph to find it\), but in general small eps values are preferable.
>
> **minPoints**: As a general rule, a minimum minPoints can be derived from a number of dimensions \(D\) in the data set, as minPoints ≥ D + 1. Larger values are usually better for data sets with noise and will form more significant clusters. The minimum value for the minPoints must be 3, but the larger the data set, the larger the minPoints value that should be chosen.

![](/assets/Screenshot 2019-08-10 at 12.32.21 PM.png)

#### Advantages:

* We don't need to specify the number of clusters
* Flexibility in the shapes and sizes of clusters
* Able to deal with noise
* Able to deal with outliers

#### Disadvantages:

* Border points are reachable from two clusters
* Faces difficulty finding clusters of varying densities

#### **Uses**

Paper:[Traffic Classification Using Clustering Algorithms](https://pages.cpsc.ucalgary.ca/~mahanti/papers/clustering.pdf)

Paper:[Anomaly detection in temperature data using dbscan algorithm](https://ieeexplore.ieee.org/abstract/document/5946052/)

