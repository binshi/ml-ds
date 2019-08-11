## Principal Component Analysis

[https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues)

[https://ourarchive.otago.ac.nz/bitstream/handle/10523/7534/OUCS-2002-12.pdf?sequence=1&isAllowed=y](https://ourarchive.otago.ac.nz/bitstream/handle/10523/7534/OUCS-2002-12.pdf?sequence=1&isAllowed=y)

Multivariate Analysis often starts out with data involving a substantial number of correlated variables. 

**Principal Component Analysis \(PCA\)** is a **dimension-reduction tool** that can be used to reduce a large set of variables to a small set that still contains most of the information in the large set.

• **Principal component analysis \(PCA\)** is a mathematical procedure that transforms a number of \(possibly\) correlated variables into a \(smaller\) number of uncorrelated variables called principal components. 

• The **first principal component** accounts for as much of the variability in the data as possible, and each succeeding component accounts for as much of the remaining variability as possible.

**Principal components analysis \(PCA\)**: PCA seeks a linear combination of variables such that the maximum variance is extracted from the variables. It then removes this variance and seeks a second linear combination which explains the maximum proportion of the remaining variance, and so on. This is called the principal axis method and results in orthogonal \(uncorrelated\) factors. PCA analyzes total \(common and unique\) variance. 

**Eigenvectors**: Principal components \(from PCA - principal components analysis\) reflect both common and unique variance of the variables and may be seen as a variance-focused approach seeking to reproduce both the total variable variance with all components and to reproduce the correlations. The principal components are linear combinations of the original variables weighted by their contribution to explaining the variance in a particular orthogonal dimension

**Eigenvalues** measure the amount of variation in the total sample accounted for by each factor.

##### **Definition of PCA**:

* Systemized way to transform input features into principal components
* Use principal components as new features
* PCs are directions in data that maximize variance\(minimize information loss\) when you project/compress down onto them
* More variance of data along a PC, higher that PC is ranked
* Max no of PCs = no of input features
* Most variance/most information --&gt; first PC --&gt; second-most variance\(without overlapping with first PC\) --&gt; second PC

##### When to use PCA

* latent features driving the patterns in data
* dimensionality reduction
  * visualize high-dimensional data
  * reduce noise
  * make other algorithms\(regression, classification\)
    * works better because fewer inputs \(eigen faces\)

**Objectives of principal component analysis** 

• PCA reduces attribute space from a larger number of variables to a smaller number of factors and as such is a "non-dependent" procedure \(that is, it does not assume a dependent variable is specified\). 

• PCA is a dimensionality reduction or data compression method. The goal is dimension reduction and there is no guarantee that the dimensions are interpretable \(a fact often not appreciated by \(amateur\) statisticians\). 

•To select a subset of variables from a larger set, based on which original variables have the highest correlations with the principal component.



