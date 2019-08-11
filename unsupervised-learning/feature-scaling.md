[https://newonlinecourses.science.psu.edu/stat505/node/54/](https://newonlinecourses.science.psu.edu/stat505/node/54/)

[http://setosa.io/ev/principal-component-analysis/](http://setosa.io/ev/principal-component-analysis/)

[https://sebastianraschka.com/Articles/2014\_about\_feature\_scaling.html](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html)

Feature scaling can vary your results a lot while using certain algorithms and have a minimal or no effect in others. To understand this, let’s look why features need to be scaled, varieties of scaling methods and when we should scale our features.

#### Why Scaling

Most of the times, your dataset will contain features highly varying in magnitudes, units and range. But since, most of the machine learning algorithms use Eucledian distance between two data points in their computations, this is a problem.

If left alone, these algorithms only take in the magnitude of features neglecting the units. The results would vary greatly between different units, 5kg and 5000gms. The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes.

To supress this effect, we need to bring all features to the same level of magnitudes. This can be acheived by scaling.

#### **How to Scale Features**

There are four common methods to perform Feature Scaling.

1. **Standardisation:**

Standardisation replaces the values by their Z scores.

![](https://miro.medium.com/max/60/1*LysCPCvg0AzQenGoarL_hQ.png?q=20)

![](https://miro.medium.com/max/133/1*LysCPCvg0AzQenGoarL_hQ.png)

This redistributes the features with their mean**μ = 0**and standard deviation**σ =1**.`sklearn.preprocessing.scale`helps us implementing standardisation in python.

**2. Mean Normalisation:**

![](https://miro.medium.com/max/60/1*fyK4gMQrfJKV5pmbXSrNbg.png?q=20)

![](https://miro.medium.com/max/197/1*fyK4gMQrfJKV5pmbXSrNbg.png)

This distribution will have values between**-1 and 1**with**μ=0**.

**Standardisation**and**Mean Normalization**can be used for algorithms that assumes zero centric data like**Principal Component Analysis\(PCA\).**

**3. Min-Max Scaling:**

![](https://miro.medium.com/max/60/1*19hq_t_NFQ6YVxMxsT0Cqg.png?q=20)

![](https://miro.medium.com/max/188/1*19hq_t_NFQ6YVxMxsT0Cqg.png)

This scaling brings the value between 0 and 1.

**4. Unit Vector:**

![](https://miro.medium.com/max/60/1*u2Up0eaer56dpmaElU3Zxw.png?q=20)

![](https://miro.medium.com/max/110/1*u2Up0eaer56dpmaElU3Zxw.png)

Scaling is done considering the whole feature vecture to be of unit length.

**Min-Max Scaling**and**Unit Vector**techniques produces values of range \[0,1\]. When dealing with features with hard boundaries this is quite useful. For example, when dealing with image data, the colors can range from only 0 to 255.

# **When to Scale** {#453c}

Rule of thumb I follow here is any algorithm that computes distance or assumes normality,**scale your features!!!**

Some examples of algorithms where feature scaling matters are:

* **k-nearest neighbors **with an Euclidean distance measure is sensitive to magnitudes and hence should be scaled for all features to weigh in equally.
* Scaling is critical, while performing
  **Principal Component Analysis\(PCA\)** PCA tries to get the features with maximum variance and the variance is high for high magnitude features. This skews the PCA towards high magnitude features.
* We can speed up **gradient descent **by scaling. This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.
* **Tree based models **are not distance based models and can handle varying ranges of features. Hence, Scaling is not required while modelling trees.
* Algorithms like **Linear Discriminant Analysis\(LDA\), Naive Bayes **are by design equipped to handle this and gives weights to the features accordingly. Performing a features scaling in these algorithms may not have much effect.

Source: [https://scikit-learn.org/stable/auto\_examples/preprocessing/plot\_scaling\_importance.html](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html)

Feature scaling through standardization \(or Z-score normalization\) can be an important preprocessing step for many machine learning algorithms. Standardization involves rescaling the features such that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one.

While many algorithms \(such as SVM, K-nearest neighbors, and logistic regression\) require features to be normalized, intuitively we can think of Principle Component Analysis \(PCA\) as being a prime example of when normalization is important. In PCA we are interested in the components that maximize the variance. If one component \(e.g. human height\) varies less than another \(e.g. weight\) because of their respective scales \(meters vs. kilos\), PCA might determine that the direction of maximal variance more closely corresponds with the ‘weight’ axis, if those features are not scaled. As a change in height of one meter can be considered much more important than the change in weight of one kilogram, this is clearly incorrect.

To illustrate this, PCA is performed comparing the use of data with[`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)applied, to unscaled data. The results are visualized and a clear difference noted. The 1st principal component in the unscaled set can be seen. It can be seen that feature \#13 dominates the direction, being a whole two orders of magnitude above the other features. This is contrasted when observing the principal component for the scaled version of the data. In the scaled version, the orders of magnitude are roughly the same across all the features.

The dataset used is the Wine Dataset available at UCI. This dataset has continuous features that are heterogeneous in scale due to differing properties that they measure \(i.e alcohol content, and malic acid\).

The transformed data is then used to train a naive Bayes classifier, and a clear difference in prediction accuracies is observed wherein the dataset which is scaled before PCA vastly outperforms the unscaled version.

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_scaling_importance_001.png "../../\_images/sphx\_glr\_plot\_scaling\_importance\_001.png")

O

