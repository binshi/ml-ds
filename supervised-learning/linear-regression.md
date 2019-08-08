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

**4.1 Stochastic Gradient Descent**

When the gradient descent is done point by point.

**4.2 Batch Gradient Descent**

When applying the squared or absolute trick to all data points, we get some values to add to the weights of the model, add them and then update the weights with the sum of those values.

**4.3 Mini Batch Gradient Descent**

In practice, neither of the previous methods is used, becaused both are slow computationally speaking. The best way to to perform a linear regression, is to split the data into many small batches. Each batch, with approximately the same number of points. Then use each batch to update the weights. This method is called Mini-Batch Gradient Descent.







####  **9. Evaluation Metrics**

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




