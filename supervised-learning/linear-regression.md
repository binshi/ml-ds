### Simple Linear Regression

Source: [https://towardsdatascience.com/supervised-learning-basics-of-linear-regression-1cbab48d0eba](https://towardsdatascience.com/supervised-learning-basics-of-linear-regression-1cbab48d0eba)

Having a set of points, the regression algorithm will model the relationship between a single feature \(explanatory variable x\) and a continuous valued response \(target variable y\).

The model will model this relationship by settting an arbitarily line and computing the distance from this line to the data points. This distance, the vertical lines, are the residuals or prediction’s errors.

The regression algorithm will keep moving the line through each iteration, trying to find the best-fitting line, in other words, the line with the minimum error.

There are several techniques to perform this task and the ultimate goal is to get the line closer to the maximum number of points.

#### **2.1 Moving The Line**

![](https://miro.medium.com/max/60/1*Yl73bpBV41F81Z1IARx8FQ.png?q=20)

![](https://miro.medium.com/max/1060/1*Yl73bpBV41F81Z1IARx8FQ.png)

##### 2.1.1 Absolute Trick

When having a point and a line, the goal is to get the line closer to this point. For achieving this task, the algorithm will use a parameter called “learning rate”. This learning rate is the number that will be multiplied to the function parameters in order to make small steps when approximating the line to the point.

In other words, the learning rate will determine the length of the distance covered in each iteration that will get the line closer to the point. It is commonly represented as the α symbol.

![](https://miro.medium.com/max/60/1*6kupC8Sy6ly5BcFWAenTEQ.png?q=20)

![](https://miro.medium.com/max/516/1*6kupC8Sy6ly5BcFWAenTEQ.png)

##### 2.1.2 Square Trick

It is based on the following premise: If there is a point closer to a line, and the distance is small, the line is moved a little distance. If it is far, the line will be moved a lot more.

![](https://miro.medium.com/max/60/1*wt2dfBichmYiaYd9ptvdlA.png?q=20)

![](https://miro.medium.com/max/794/1*wt2dfBichmYiaYd9ptvdlA.png)

#### **3. Gradient Descent**

Let us say that we have a set of points and we want to develop an algorithm that will find the line that best fits this set of points. The error, as stated before, will be the distance from the line to the points.

The line is moved and the error is computed. This process is repeated over and over again, reducing the error a little bit each time, until the perfect line is achieved. This perfect line will be the one with the smallest error.To minimize this error, we will use the gradient descent method.

Gradient descent is a method that, for each step, will take a look at the different directions which the line could be moved to reduce the error and will take the action that reduce most this error.

The definition of the gradient from Wikipedia:

> The gradient of an escalar field \(f\), is a vectorial field. When it is evaluated in a generic point of the domain of f, it indicates the direction of quicker variance of the field f.

So the gradient descent will take a step in the direction of the negative gradient.

![](https://miro.medium.com/proxy/0*rBQI7uBhBKE8KT-X.png "Resultado de imagen de gradient descent")

When this algorithm has taken sufficient steps, it will eventually get to a local or global minimum, if the learning rate is set to an adequate value. This is a very important detail, because, if the learning rate is too high, the algorithm will keep missing the minimum, as it will take steps too large. And if this learning rate is too low, it will take infinite time to get to the point.

![](https://miro.medium.com/proxy/0*QwE8M4MupSdqA3M4.png "Resultado de imagen de gradient descent")

#### **4. Gradient Descent Methods**

At this point, it seems that we've seen two ways of doing linear regression.

* By applying the squared \(or absolute\) trick at every point in our data _one by one_, and repeating this process many times.
* By applying the squared \(or absolute\) trick at every point in our data _all at the same time_, and repeating this process many times.

More specifically, the squared \(or absolute\) trick, when applied to a point, gives us some values to add to the weights of the model. We can add these values, update our weights, and then apply the squared \(or absolute\) trick on the next point. Or we can calculate these values for all the points, add them, and then update the weights with the sum of these values.

The latter is called _batch gradient descent_. The former is called_stochastic gradient descent_.

**4.1 Stochastic Gradient Descent**

When the gradient descent is done point by point.

**4.2 Batch Gradient Descent**

When applying the squared or absolute trick to all data points, we get some values to add to the weights of the model, add them and then update the weights with the sum of those values.

**4.3 Mini Batch Gradient Descent**

In practice, neither of the previous methods is used, becaused both are slow computationally speaking. The best way to to perform a linear regression, is to split the data into many small batches. Each batch, with approximately the same number of points. Then use each batch to update the weights. This method is called Mini-Batch Gradient Descent.

#### **5. Higher Dimensions**

When we have one input column and one output column, we are facing a two-dimensional problem and the regression is a line. The prediction will be a constant by the independent variable plus other constant.

If we have more input columns, it means that there are more dimensions and the output will not be a line anymore, but planes or hyperplanes \(depending on the number of dimensions\).

![](https://miro.medium.com/max/30/1*nTReJPwY8ToEX1CSonTWAg.png?q=20)

![](https://miro.medium.com/max/505/1*nTReJPwY8ToEX1CSonTWAg.png)

#### **6. Multiple Linear Regression**

Independent variables are also known as predictors, which are variables we look at to make predictions about other variables. Variables we are trying to predict are known as dependant variables.

When the outcome we are trying to predict depends on more than one variable, we can make a more complicated model that takes this higher dimensionality into account. As long as they are relevant to the problem faced, using more predictor variables can help us to get a better prediction.

As seen before, the following image shows a simple linear regression:

![](https://miro.medium.com/max/30/1*LuwSoQw01wd3u9ruliSw6w.png?q=20)

![](https://miro.medium.com/max/603/1*LuwSoQw01wd3u9ruliSw6w.png)

And the following picture shows a fitted hyperplane of multiple linear regression with two features.

![](https://miro.medium.com/max/30/1*uqZFeQ0MlZ8nhaX2c4bAWw.png?q=20)

![](https://miro.medium.com/max/464/1*uqZFeQ0MlZ8nhaX2c4bAWw.png)

As we add more predictors, we add more dimensions to the problem and it becomes harder to visualize it, but the core of the process remains the same.

#### **7. Linear Regression Warnings**

Linear regression comes with a set of assumptions and we should take into account that it is not the best model for every situation.

**a\) Linear Regression works best when data is linear:**

It produces a straight line from the training data. If the relastionship in the training data is not really linear, wewill need to either make adjustments \(transforming training data\), add features or use other model.

**b\) Linear Regression is sensitive to outliers:**

Linear regression tries to fit a best line among the training data. If the dataset has some outlying extreme values that do not fit a general pattern, linear regression models can be heavily impacted by the presence of outliers. We will have to watch out for these outliers and normally remove then.

One common method to deal with outliers is to use and alternative method of regression which specially robust against this extreme values. This method is called RANdom Sample Consensus \(RNASAC\) algorithm, which fits the model to the inliers subset of data. The algorithm performs the following steps:

* It selects a random number of samples to be inliers and fit the model.
* It tests all other data points against the fitted model and add the ones that fall within the user-chosen value.
* Repeats the fitting of the model with the new points.
* Compute the error of the fitted model against the inliers.
* End the algorithm if the perfomance meets a certain user-defined threshold or a number of iterations is reached. Otherwise, it goes back to the first step.

  
8**. Polynomial Regression**

Polynomial regression is a special case of multiple linear[r](https://en.wikipedia.org/wiki/Regression_analysis)egression analysis in which the relationship between the independetn variable_x_and the dependent variable_y_is modelled as an_n_th degree polynomial in_x_. In other words, when our data distribution is more complex than a linear one, and we generate a curve using linear models to fit non-linear data.

The independent \(or explanatory\) variables resulting from the polynomial expansion of the predictor variables are known as higher-degree terms. It has been used to describe nonlinear phenomena such as the growth rate of tissues and the progression of disease epidemics.

![](https://miro.medium.com/max/30/1*7rsG2HMIuNrtRhRWSCZvJQ.png?q=20)

![](https://miro.medium.com/max/582/1*7rsG2HMIuNrtRhRWSCZvJQ.png)

#### 9**. Regularization**

Regularization, is a widely used method to deal with overfitting. It is done mainly by the following techniques:

1. Reducing the model’s size: Reducing the number of learnable parameters in the model, and with them its learning capacity. The goal is to get to a sweet spot between too much and not enough learning capacity. Unfortunately, there aren’t any magical formulas to determine this balance, it must be tested and evaluated by setting different number of parameters and observing its performance.
2. Adding weight regularization: In general, the simpler the model the better. As long it can learn well, a simpler model is much less likely to overfit. A common way to achieve this, is to constraint the complexity of the network by forcing its weights to only take small values, regularizating the distribution of weight values. This is done by adding to the loss function of the network a cost associated with having large weights. The cost comes in two ways:

* L1 regularization: The cost is proportional to the absolute value of the weight coefficients \(L1 norm of the weights\).
* L2 regularization: The cost is proportional to the square of the value of the weight coefficients \(l2 norm of the weights\)

![](https://miro.medium.com/max/30/0*ITv81egIxcJdJEbk?q=20)

![](https://miro.medium.com/max/550/0*ITv81egIxcJdJEbk)

To decide which of them to apply to our model, is recommended to keep the following information in mind and take into account the nature of our problem:

![](https://miro.medium.com/max/30/0*6aQbHz4Kx8PWJ7sa?q=20)

![](https://miro.medium.com/max/389/0*6aQbHz4Kx8PWJ7sa)

* The λ Parameter: It is the computed error by regularization. If we have a large λ, then we are punishing complexity and will end up with a simpler model. If we have a small λ we will end up with a complex model.

#### **9. Evaluation Metrics**

In order to keep track of how well our model is performing, we need to set up some evaluation metrics. This evaluatioin metric is the error computed from the generated line \(or hyperplane\) to the real points and will be the function to minimize by the gradient descent .

Some of the most common regression evaluation metrics are:

**9.1 Mean Absolute Error:**

![](https://miro.medium.com/max/30/1*NQu_7ukauHIiofUKpN1hmQ.png?q=20)

![](https://miro.medium.com/max/535/1*NQu_7ukauHIiofUKpN1hmQ.png)

Mean Absolute Error or MAE, is the average of the absolute difference between the real data points and the predicted outcome. If we take this as the strategy to follow, each step of the gradient descent would reduce the MAE.

![](https://miro.medium.com/max/30/1*fgyZdSNiLbmGZAH_2f5UQw.png?q=20)

![](https://miro.medium.com/max/373/1*fgyZdSNiLbmGZAH_2f5UQw.png)

**9.2 Mean Square Error:**

![](https://miro.medium.com/max/30/1*L9m9c8MIo0fFXZ1MZnxBxA.png?q=20)

![](https://miro.medium.com/max/520/1*L9m9c8MIo0fFXZ1MZnxBxA.png)

Mean Square Error or MSE, is the average of the squared difference between the real data points and the predicted outcome. This method penalizes more the bigger the distance is, and it is the standard in regression problems.

If we take this as the strategy to follow, each step of the gradient descent would reduce the MSE. This will be the preferred method to compute the best-fitting line, and it is also called Ordinary Least Squares or OLS.

![](https://miro.medium.com/max/30/1*4gP8H_1pMpGpGzv5-3NjYg.png?q=20)

![](https://miro.medium.com/max/373/1*4gP8H_1pMpGpGzv5-3NjYg.png)

**9.3 Root Mean Squared Error**

The Root Mean Squared Error, or RMSE, is the root of the mean of the squared errors, and it is the most popular evaluation metric for determining the performance of regression models, as the root yields the same units as the y.

![](https://miro.medium.com/max/30/1*zlL2nNjWznwrToDAnInt3Q.png?q=20)

![](https://miro.medium.com/max/432/1*zlL2nNjWznwrToDAnInt3Q.png)

**9.4 Coefficient of Determination or R²**

The coefficient of determination can be understood as a standardize version of the MSE, which provides a better interpretability of the performance of the model.

Technically, the R² is the fraction of the response variance that is captured by the model, in other words it is the variance of the response. It is defined as:

![](https://miro.medium.com/max/30/0*_j3Q5--UHQz97P5G.png?q=20)

![](https://miro.medium.com/max/562/0*_j3Q5--UHQz97P5G.png)

![](/assets/Screenshot 2019-08-08 at 5.40.31 PM.png)

