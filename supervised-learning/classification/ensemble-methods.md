## Ensemble Methods

An ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model. 

#### Bagging\(Bootstrap Aggregating\):

Pick the most frequent answer and use it to classify. Pick a subset of data and train models.

Bootstrap Aggregation is a general procedure that can be used to reduce the variance for those algorithm that have high variance. An algorithm that has high variance are decision trees, like classification and regression trees \(CART\).

Decision trees are sensitive to the specific data on which they are trained. If the training data is changed \(e.g. a tree is trained on a subset of the training data\) the resulting decision tree can be quite different and in turn the predictions can be quite different.

Bagging is the application of the Bootstrap procedure to a high-variance machine learning algorithm, typically decision trees.

Let’s assume we have a sample dataset of 1000 instances \(x\) and we are using the CART algorithm. Bagging of the CART algorithm would work as follows.

1. Create many \(e.g. 100\) random sub-samples of our dataset with replacement.
2. Train a CART model on each sample.
3. Given a new dataset, calculate the average prediction from each model.

For example, if we had 5 bagged decision trees that made the following class predictions for a in input sample: blue, blue, red, blue and red, we would take the most frequent class and predict blue.

When bagging with decision trees, we are less concerned about individual trees overfitting the training data. For this reason and for efficiency, the individual decision trees are grown deep \(e.g. few training samples at each leaf-node of the tree\) and the trees are not pruned. These trees will have both high variance and low bias. These are important characterize of sub-models when combining predictions using bagging.

The only parameters when bagging decision trees is the number of samples and hence the number of trees to include. This can be chosen by increasing the number of trees on run after run until the accuracy begins to stop showing improvement \(e.g. on a cross validation test harness\). Very large numbers of models may take a long time to prepare, but will not overfit the training data.

#### Boosting:

[http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)

[https://www.quora.com/What-is-an-intuitive-explanation-of-Gradient-Boosting](https://www.quora.com/What-is-an-intuitive-explanation-of-Gradient-Boosting)

Boosting is an approach to machine learning based on the idea of creating a highly accurate prediction rule by combining many relatively weak and inaccurate rules

for a learned classifier to be effective and accurate in its predictions, it should meet three conditions: \(1\) it should have been trained on “enough” training examples; \(2\) it should provide a good fit to those training examples \(usually meaning that it should have low training error\); and \(3\) it should be “simple.” This last condition, our expectation that simpler rules are better, is often referred to as Occam’s razor.

##### Adaboost: [http://rob.schapire.net/papers/explaining-adaboost.pdf](http://rob.schapire.net/papers/explaining-adaboost.pdf)

Fit the model as accurately as possible. Pick the next model and use it on the same data by increasing the weights of the misclassified points from first model which will result in the moodel classifying the misclassified points correctly and so on.

![](/assets/Screenshot 2019-08-09 at 7.53.13 PM.png)

