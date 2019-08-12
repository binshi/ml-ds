## Random Projection

#### Johnson-Lindenstrauss Lemma

**If points in vector space are projected onto a randomly selected subspace of suitably high dimensions, then the distances between the points are approximately preserved.**

* In RP, a higher dimensional data is projected onto a lower-dimensional subspace using a random matrix whose columns have unit length.
* RP is computationally efficient, yet accurate enough for this purpose as it does not introduce a significant distortion in the data.
* It is not sensitive to impulse noise. So RP is promising alternative to some existing methods in noise reduction \(like mean filtering\) too.
* The original d-dimensional data is projected to a k-dimensional \(k&lt;&lt;d\) through the origin, using a random k∗d matrix R whose columns have unit lengths. It is given by

![](/assets/Screenshot 2019-08-12 at 7.34.54 AM.png)**Complexity**

* * Forming a random matrix R and projecting d \* N data matrix X into k dimensions is of the order
    **O\(dkN\)**
  * If X is a sparse matrix with c non-zero values per column, then the complexity is
    **O\(ckN\)**
* Theoretically, equation \(1\) is not a projection because R is generally not orthogonal. A linear mapping like \(1\) can cause significant distortion in data if R is not orthogonal.**Orthogonalizing R is computationally expensive.**

* Instead of orthogonalizing, RP relies on the result presented by Hecht-Neilsen i.e.**In a high dimensional space, there exists a much larger number of almost orthogonal than orthogonal directions.**Thus the vectors with random directions might be sufficiently close to orthogonal, and equivalentlyRTRRTRwould approximate an identity matrix.

* Experimental results show the mean squared difference betweenRTRRTRand identity matrix is around**1/k per element**.

Paper:[Random projection in dimensionality reduction: Applications to image and text data](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.8124&rep=rep1&type=pdf)

This paper examines using Random Projection to reduce the dimensionality of image and text data. It shows how Random Projection proves to be a computationally simple method of dimensionality reduction, while still preserving the similarities of data vectors to a high degree. The paper shows this on real-world datasets including noisy and noiseless images of natural scenes, and text documents from a newsgroup corpus.

Paper:[Random Projections for k-means Clustering](https://papers.nips.cc/paper/3901-random-projections-for-k-means-clustering.pdf)

This paper uses Random Projection as an efficient dimensionality reduction step before conducting k-means clustering on a dataset of 400 face images of dimensions 64 × 64.

When random projections can be better than PCA:

* Data is so high dimensional that it is too expensive to compute principal components directly 
* You do not have access to all the data at once, as in data streaming 
* Data is approximately low-dimensional, but not near a linear subspace



