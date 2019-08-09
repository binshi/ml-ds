## Ensemble Methods

#### Bagging\(Bootstrap Aggregating\):

Pick the most frequent answer and use it to classify. Pick a subset of data and train models

#### Boosting:

[http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)

[https://www.quora.com/What-is-an-intuitive-explanation-of-Gradient-Boosting](https://www.quora.com/What-is-an-intuitive-explanation-of-Gradient-Boosting)

Boosting is an approach to machine learning based on the idea of creating a highly accurate prediction rule by combining many relatively weak and inaccurate rules

for a learned classifier to be effective and accurate in its predictions, it should meet three conditions: \(1\) it should have been trained on “enough” training examples; \(2\) it should provide a good fit to those training examples \(usually meaning that it should have low training error\); and \(3\) it should be “simple.” This last condition, our expectation that simpler rules are better, is often referred to as Occam’s razor.

##### Adaboost: [http://rob.schapire.net/papers/explaining-adaboost.pdf](http://rob.schapire.net/papers/explaining-adaboost.pdf)

Fit the model as accurately as possible. Pick the next model and use it on the same data by increasing the weights of the misclassified points from first model which will result in the moodel classifying the misclassified points correctly and so on.

![](/assets/Screenshot 2019-08-09 at 7.53.13 PM.png)

