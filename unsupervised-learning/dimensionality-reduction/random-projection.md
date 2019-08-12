## Random Projection

Paper:[Random projection in dimensionality reduction: Applications to image and text data](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.8124&rep=rep1&type=pdf)

This paper examines using Random Projection to reduce the dimensionality of image and text data. It shows how Random Projection proves to be a computationally simple method of dimensionality reduction, while still preserving the similarities of data vectors to a high degree. The paper shows this on real-world datasets including noisy and noiseless images of natural scenes, and text documents from a newsgroup corpus.

Paper:[Random Projections for k-means Clustering](https://papers.nips.cc/paper/3901-random-projections-for-k-means-clustering.pdf)

This paper uses Random Projection as an efficient dimensionality reduction step before conducting k-means clustering on a dataset of 400 face images of dimensions 64 × 64.



When random projections can be better than PCA: 

* Data is so high dimensional that it is too expensive to compute principal components directly 
* You do not have access to all the data at once, as in data streaming 
* Data is approximately low-dimensional, but not near a linear subspace



