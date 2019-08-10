# Hierarchical Clustering

Hierarchiical

In **single linkage** hierarchical clustering, the distance between two clusters is defined as the shortest distance between two points in each cluster. Single Linkage hierarchical clustering is more prone to result in elongated shapes that are not necessarily compact or circular

In **complete linkage** hierarchical clustering, the distance between two clusters is defined as the longest distance between two points in each cluster.

In **average linkage** hierarchical clustering, the distance between two clusters is defined as the average distance between each point in one cluster to every point in the other cluster.

**Ward's method**: Hierarchical clustering works by merging clusters that lead to the least increase in variance in the clusters after merging

![](/assets/Screenshot 2019-08-10 at 11.35.39 AM.png)![](/assets/Screenshot 2019-08-10 at 11.38.15 AM.png)![](/assets/Screenshot 2019-08-10 at 11.40.02 AM.png)![](/assets/Screenshot 2019-08-10 at 11.48.06 AM.png)

![](/assets/Screenshot 2019-08-10 at 11.49.57 AM.png)

![](/assets/Screenshot 2019-08-10 at 11.51.04 AM.png)

##### Advantages

* Resulting hierarchical representation can be very informative
* Provides an addiitional ability to visualize
* Especially potent when the dataset contains real hierarchical relationships \(e.g. Evolutionary biology\) 

#####  Disadvantages

* Sensitive to noise and outliers
* Computationally intensive O\(N\*\*2\)

**Uses:**

Paper:[Using Hierarchical Clustering of Secreted Protein Families to Classify and Rank Candidate Effectors of Rust Fungi](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029847)

Paper:[Association between composition of the human gastrointestinal microbiome and development of fatty liver with choline deficiency](https://www.ncbi.nlm.nih.gov/pubmed/21129376)

