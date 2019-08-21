Source: [https://towardsdatascience.com/supervised-learning-basics-of-classification-and-main-algorithms-c16b06806cd3](https://towardsdatascience.com/supervised-learning-basics-of-classification-and-main-algorithms-c16b06806cd3)

[https://github.com/ctufts/Cheat\_Sheets/wiki/Classification-Model-Pros-and-Cons](https://github.com/ctufts/Cheat_Sheets/wiki/Classification-Model-Pros-and-Cons)

Classification is a subcategory of supervised learning where the goal is to predict the categorical class labels \(discrete, unoredered values, group membership\) of new instances based on past observations.

There are two main types of classification problems:

* Binary classification: The typical example is e-mail spam detection, which each e-mail is spam → 1 spam; or isn’t → 0.
* Multi-class classification: Like handwritten character recognition \(where classes go from 0 to 9\).

The following example is very representative to explain binary classification:

There are 2 classes, circles and crosses, and 2 features, X1 and X2. The model is able to find the relationship between the features of each data point and its class, and to set a boundary line between them, so when provided with new data, it can estimate the class where it belongs, given its features.

![](https://miro.medium.com/proxy/1*fBjniQPOKigqxYSKEumXoA.png)

In this case, the new data point falls into the circle subspace and, therefore, the model will predict its class to be a circle.

**Different Classes**

It is important to note that not every classification models will be useful to separate properly different classes from a dataset. Some algorithms, like the perceptron, \(which is based on basic artificial neural networks\) if the classes can’t be separated by a linear decision boundary, will not converge when learning the model’s weights, .

Some of the most typical casses are represented in the following picture:

![](https://miro.medium.com/max/30/1*q3oqM2j4HOs6ntxZ_OakFw.png?q=20)

![](https://miro.medium.com/max/700/1*q3oqM2j4HOs6ntxZ_OakFw.png)

So, the task of selecting an appropiate algorithm became of paramount importance in classification problems, and this will be one of the main topics that will be discussed throughout the article.

**Classification in Practice**

In practice, it is always recommended to try and compare different algorithm’s performance, in order to choose the most appropiate to tackle the problem. This performance will be very influenced by the data available, number of features and samples, the different classes and whether they are linearly separable or not.

To remind ourselves the six main steps to do in the development of a machine learning model:

1. Collect data.
2. Choose a measure of success.
3. Setting an evaluation protocol.
4. Preparing the data
5. Developing a benchmark model
6. Developing a better model and tunning its hyperparamters

Next, well proceed to explore the different classification algorithms and learn which one is more suitable to perform each task.

### Logistic Regression

One of the main problems in classification problems occurs when the algorithm never converges in the weight’s updating, while being trained.

This occurs when the classes aren’t perfectly linear separable. So, to tackle binary classification problems , the Logistic Regression is one of the most used algorithms.

Logistic regression is a simple but powerful classification algorithm \(despite of its name\). It works very well on linearly separable classes and can be extended to multiclass classification, via the OvR technique.

**Odds Ratio**

The odds ratio is one important concept in order to understand the idea behind logistic regression.

The odds ratio is the probability that a certain event will occur. It can be written as:

![](https://miro.medium.com/max/30/1*rOHJHkL5fyiNAlYsX7nyWw.png?q=20)

![](https://miro.medium.com/max/246/1*rOHJHkL5fyiNAlYsX7nyWw.png)

Where P stands for the probability of the positive event \(the one that we are trying to predict\).

Derived from this we can define the logit function.

**Logit Function**

![](https://miro.medium.com/max/30/1*fHXwvuv7wJCIL-g0_FBicw.png?q=20)

![](https://miro.medium.com/max/245/1*fHXwvuv7wJCIL-g0_FBicw.png)

The logit function is simply the logarithm of the odds ratio \(log-odds\). This function takes as am input values in the range \[0,1\] and trasnforms them to values over the entire real-number range \[-∞,∞\].

We will use it to express linear relationships between feature values and the log-odds.

![](https://miro.medium.com/max/30/1*HXOwPxofriTntkZV2cFNFg.png?q=20)

![](https://miro.medium.com/max/574/1*HXOwPxofriTntkZV2cFNFg.png)

Where P\(y=1\|x\) is the conditional probability that a particular sample belongs to class 1 given its features x.

Our true motivation behind this is to predict the probabilty that a sample belongs to a certain class. This is the inverse of the logit function, and is frequently called the sigmoid function.

**The Sigmoid Function**

The formula of the sigmoid function is:

![](https://miro.medium.com/max/30/1*YQjrzvGluj3lGIKBgkioug.png?q=20)

![](https://miro.medium.com/max/230/1*YQjrzvGluj3lGIKBgkioug.png)

Z is the net input, which is the linear combination of weights and sample features and can be calculated as:

![](https://miro.medium.com/max/30/1*26fVtw8fQTxmw_PGzug73A.png?q=20)

![](https://miro.medium.com/max/329/1*26fVtw8fQTxmw_PGzug73A.png)

When it is represented in a graphic, it adopts the following shape:

![](https://miro.medium.com/max/30/1*tc_yXtKbmj-luCNO14sNHw.png?q=20)

![](https://miro.medium.com/max/576/1*tc_yXtKbmj-luCNO14sNHw.png)

We can see that there are two limits both in the Ø\(z\) equal to 1 and 0 values . It means that the function approaches to one if the z tends to infinity and approaches to zero if the z tends to minus infinity.

So it takes real values and transforms them to the \[0,1\] range, with an intercept in Ø\(z\) = 0.5.

![](https://miro.medium.com/max/30/1*Spq9_ksfgMbMB3VNZUz1AQ.png?q=20)

![](https://miro.medium.com/max/697/1*Spq9_ksfgMbMB3VNZUz1AQ.png)

In summary, this is what the logistic regression model does while being trained. The output of the sigmoid function is interpreted as the probability of a certan sample to belong to class 1, given its features x parametrized by the weights w as, _Ø\(z\) =P\(y=1\|x; w\)._

The predicted probability can be converted into a binary outcome by unit step function \(a quantizier\):

![](https://miro.medium.com/max/30/1*77OkXusnt6sV-VL8iSOpYg.png?q=20)

![](https://miro.medium.com/max/222/1*77OkXusnt6sV-VL8iSOpYg.png)

Looking at the previous sigmoid graph, the equivalence will be:

![](https://miro.medium.com/max/30/1*A9PFnwFfvZZreG19gRsnyw.png?q=20)

![](https://miro.medium.com/max/221/1*A9PFnwFfvZZreG19gRsnyw.png)

And this is one of the main reasons why the logistic regression algorithm is so popular, because it returns the probability \(as the value between 0 and 1\) of a certain sample of belonging to a particular class.

This is extremely useful in cases like weather forecasting, where you don’t only would like if its going to be rainny day but also the chance of getting rain. Or to predict the chance of a patient on having a certain disease.

### Support Vector Machines \(SVM\)

This algorithm can be considered as an extension of [the perceptron algorithm](https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975). In SVM, The optimization objective is to set a decision line that separates the classes by maximizing the marging between this line and the sample points that are closest to this hyperplane. This points are called support vectors.

![](https://miro.medium.com/max/24/1*5GNKb6FtV8RKgzEKi9ukIw.png?q=20)

![](https://miro.medium.com/max/499/1*5GNKb6FtV8RKgzEKi9ukIw.png)

**Maximum Margin**

To set the maximum margins, there are added two parallel lines \(margins\) and we try to maximize their distances to the original decision line. We will take into account the misclassified points \(errors\) and the ones between the margins and the line.

Normally, decision lines with large margins tend to have a lower generalization error. On the other hand, models with small margins tend to be less prone to overfitting.

The maximization function \(marging of error\) is calculated as follows:

1. The positive and negative hyperplanes can be expressed as:

![](https://miro.medium.com/max/30/1*SUhsVG6J1mtd3KuyuTir-w.png?q=20)

![](https://miro.medium.com/max/358/1*SUhsVG6J1mtd3KuyuTir-w.png)

1. Substracting \(1\) and \(2\) from each other:

![](https://miro.medium.com/max/30/1*wbFs5654qdOYdUMFG3DZ_w.png?q=20)

![](https://miro.medium.com/max/352/1*wbFs5654qdOYdUMFG3DZ_w.png)

1. Normalizing the previous equation by the length of the _w_ vector, whicch is:

![](https://miro.medium.com/max/30/1*2hnA3MlTZHG9bIJsDEZ7LQ.png?q=20)

![](https://miro.medium.com/max/263/1*2hnA3MlTZHG9bIJsDEZ7LQ.png)

1. We arrive to the margin error equation:

![](https://miro.medium.com/max/30/1*4Gb8ZXovU0-NjrnxZy_ULg.png?q=20)

![](https://miro.medium.com/max/379/1*4Gb8ZXovU0-NjrnxZy_ULg.png)

Which left side is interpreted as the distance between the positive and negative hyperplanes, in other words, the marging we are trying to maximize.

In practice, it is easier to minimize the reciprocal term, which can be solved with quadratic programming.

![](https://miro.medium.com/max/30/1*-0AAj0S_yp2wcRbj4dNJBA.png?q=20)

![](https://miro.medium.com/max/192/1*-0AAj0S_yp2wcRbj4dNJBA.png)

**Slack Variables: Dealing with Nonlinearly Separable Classes**

The slack variable ξ it is used with soft-marging classification. It was motivated by the need to relax the linear constraints, in order to allow convergence of the optimization process when dealing with non linearly separable data.

It is added to the linear constraints and the new minimization function becomes:

![](https://miro.medium.com/max/30/1*LckrO_x7nDHn8pxlOO5zKw.png?q=20)

![](https://miro.medium.com/max/390/1*LckrO_x7nDHn8pxlOO5zKw.png)

![](https://miro.medium.com/max/30/1*fx02s7M5xu-q6yBq1l16VA.png?q=20)

![](https://miro.medium.com/max/476/1*fx02s7M5xu-q6yBq1l16VA.png)

By varying the C variable of the function, we can control the penalty for misclassification, and control the width of the margin. Tunning in this way the bias-variance trade-off.

The best boundary line will depend on the problem that we are trying to solve. If we are facing a medical problem , we don’t want any mistakes. Whereas we could deal with some mistakes in the case of facing other type of problem.

This is when the C value comes into place.

* Large C values correspond to large error penalties.
* Smaller C values imply that we are less strict about misclassification errors.

![](https://miro.medium.com/max/30/1*yt4a0-dV10crYHVEpvGbHg.png?q=20)

![](https://miro.medium.com/max/700/1*yt4a0-dV10crYHVEpvGbHg.png)

**Polynomial Kernel SVM**

SVM are a very popular branch of algorithms because it can be used to solve nonlinear classification problems. This is done by a method called kernelizing.

The basic idea of using kernels when dealing with non linear combinantions of the original features is to project them onto a higher dimensional space via a mapping function Ø, so the data becomes linearly separable.

Intuitively, the original data set is transformed to a higher dimensional one and then a projection is applied to make the classes separable.

Then the algorithm is applied, the classes are split and it is applied the inverse of the projection function to come back to the original distribution of the data.

![](https://miro.medium.com/max/30/1*mxxko6is861MOJ51ANCklw.png?q=20)

![](https://miro.medium.com/max/700/1*mxxko6is861MOJ51ANCklw.png)

**RBF Kernel**

Also used when dealing with non linearly separable data, like the following:

![](https://miro.medium.com/max/30/1*XCGBPvrA9JJrF6gFZOMuwQ.png?q=20)

![](https://miro.medium.com/max/700/1*XCGBPvrA9JJrF6gFZOMuwQ.png)

![](https://miro.medium.com/max/30/1*0_C0sgpKZTTsPl8Qk_NTiw.png?q=20)

![](https://miro.medium.com/max/700/1*0_C0sgpKZTTsPl8Qk_NTiw.png)

The idea is to locate mountains and valleys that will coincide with each class. And project the lines that separates two classes and cut also the mountain-valley line, to the original line:

![](https://miro.medium.com/max/30/1*xqOV6SXXO5ELdVicngeY6g.png?q=20)

![](https://miro.medium.com/max/700/1*xqOV6SXXO5ELdVicngeY6g.png)

The projections will effectively separate each class on our original dataset.

To vary the width of the mountains and valleys, we will use the gamma parameter γ.

The gamma parameter is a hyperparameter that we tune during training and for:

* Large γ values → will yield narrow mountains/valleys that will tend to overfit.
* Small values → will yield wide mountains/valleys that will tend to underfit.

This parameter comes from the formula of the normal distribution:

![](https://miro.medium.com/max/30/1*7H6ddQZwwkz2B85lnDiN9g.png?q=20)

![](https://miro.medium.com/max/700/1*7H6ddQZwwkz2B85lnDiN9g.png)

### Decision Trees Algorithms

Decision trees algorithms break down the dataset by making questions until they have narrowed data enough to make a prediction.

This is an example of a decision tree for deciding if you lend someone your car:

![](https://miro.medium.com/max/30/1*s6bohT7lyP4MZCNBSrJHgg.png?q=20)

![](https://miro.medium.com/max/700/1*s6bohT7lyP4MZCNBSrJHgg.png)

Based on the features of the training set, the decision tree learns a series of questions to infer the class labels of the samples.

The starting node is called the tree root, and the algorithm will split the dataset on the feature that contains the maximum Information Gain iteratively, until the leaves \(the final nodes\) are pure.

**Decision Trees Hyperparameters**

a\) Maximum Depth:

Maximum Depth is the largest length from the root to the leaf. Large depth can cause overfitting and small depth can cause underfitting.To avoid overfitting, we’ll prune the decision tree by setting a hyperparameter with the maximal depth.

![](https://miro.medium.com/max/30/1*srVqKi5KJiu5sYAF1agoxA.png?q=20)

![](https://miro.medium.com/max/406/1*srVqKi5KJiu5sYAF1agoxA.png)

b\) Maximum Number of Samples:

When splitting a node, one could run into the problem of having 99 samples inone of the splits and 1 in the other one. This will be a waste of resources, to avoid it, we can set a maximum for the number of samples we allow for each leaf. It can be specified as an integer or as a float.

A small minimum number of samples will lead to overfitting, whereas a large number will lead to underfitting.

c\) Minimum Number of Samples:

Analog to the previous one, but with minimum values.

d\) Maximum Number of Features:

Very often, we will have to many features to buid a tree. In every split, we have to check the entire dataset on each one of the features, which can be very expensive.

A solution to this problem is to limit the number of features that one looks for in each split. If this number is high enough, we are likely to find a good feature among the ones we look for \(although it may not be the perfect one\). However, if it is not as large as the number of features, it will speed up the calculations signficantly.

# Random Forests {#f97a}

If we have a dataset with a lot of features \(columns\), the decision tree Algorithm usually tend to overfit, overcomplicating the model and the learning process.

We can solve this issue by selecting each colum randomly and making decision trees for each batch of columns.

![](https://miro.medium.com/proxy/1*i0o8mjFfCn-uD79-F1Cqkw.png "Resultado de imagen de random forests")

Original picture from

[this article](https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d)

So the idea is to develop an ensemble learning algorithm which will combine a number of weaker models to build a more robust one.

The algorithm will perform the following steps:

* Drawing of a random bootstrap sample of size n.
* Growing a decision tree from the bootstrap sample. At each node: Therewill be randomly selected d features without replacement and the node will be splitted maximizing the information gain.
* The previous process willbe repeated k times.
* Agreggating the predicition done by each tree, asigning the class label by majority vote.

The main advantage of this method is that we usually won’t need to prune the random forest \(since the model is quite robust to noise\). However, they are much less interpretable than decision trees.

The only hyperparameter that we will need to tune is the _K_ number of trees. Normally, the larger _K_ is, the better that the model will perform, but it will increase drastically the computational cost.

# K-Nearest Neighbors \(KNN\) {#4ba5}

K-nearest neighbor algorithms or KNN, belongs to a special type of machine learning models that are frequently called ‘lazy algorithms’. They receive this name because they do not learn how to discriminate the dataset with an optimized function, but memorize the dataset instead. The name ‘lazy algorithm’ also refers to the kind of algorithms called nonparametric. These are instance-based algorithms, they are characterized by memorizing the training dataset, and lazy learning is a specific case of these algorithms, asociated with zero computational cost during the learning.

**The algorithm**

The overall process that the algorithm follows is:

1. Choosing the number of k and the distance metric.
2. Finding the k nearest neighbor of the sample to classify
3. Assigning the class label by majority vote

![](https://miro.medium.com/proxy/0*Sk18h9op6uK9EpT8. "Resultado de imagen de k-nearest neighbors")

The algorithm find those k samples that are closest to the point to classify, basing its predictions on the distance metric.

The main advantage is that as it is a memory-based algorithm, it adapts to new training data. The down-side is that the computational cost increases linearly with the size of the training data.

**Things to Take Into Account**

* In the case that the algorithm faces a tie, it will prefer the neighbors with a closer distance to the classification sample. In the case of having a similar distance, then KNN will choose the class label that comes first in the dataset.
* It is fundamental to choose the right k value in order to have a good balance between over and underfitting.
* It is also crucial to stablish an appropiate distance metric. Usually, it is used the ‘Minkowski’ distance, which a generalization of the Eucledian and Manhattan distance.This distance is defined as follows:

![](https://miro.medium.com/max/30/1*OWL0d7F2b1NovCUZKnafNA.png?q=20)

![](https://miro.medium.com/max/274/1*OWL0d7F2b1NovCUZKnafNA.png)

# Conclusion {#bb32}

In this article, we have learned the basis of classification and the principal algorithms and how to tune them according the problem faced.

