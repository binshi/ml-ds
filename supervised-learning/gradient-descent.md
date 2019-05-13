[https://blog.goodaudience.com/gradient-descent-for-linear-regression-explained-7c60bc414bdd](https://blog.goodaudience.com/gradient-descent-for-linear-regression-explained-7c60bc414bdd)

[http://mccormickml.com/2014/03/04/gradient-descent-derivation/](http://mccormickml.com/2014/03/04/gradient-descent-derivation/)

[https://medium.com/datadriveninvestor/gradient-descent-5a13f385d403](https://medium.com/datadriveninvestor/gradient-descent-5a13f385d403)

https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f

https://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html

#### Batch Gradient Descent {#f213}

In batch gradient we**use the entire dataset to compute the gradient of the cost function for each iteration of the gradient descent and then update the weights. **Since we use the entire dataset to compute the gradient convergence is slow.If the dataset is huge and contains millions or billions of data points then it is memory as well as computationally intensive.

**Advantages of Batch Gradient Descent**

* Theoretical analysis of weights and convergence rates are easy to understand

**Disadvantages of Stochastic Gradient Descent**

* Perform redundant computation for the same training example for large datasets
* Can be very slow and intractable as large datasets may not fit in the memory
* As we take the entire dataset for computation we can update the weights of the model for the new data

#### Stochastic Gradient descent {#8e93}

In stochastic gradient descent we**use a single datapoint or example to calculate the gradient and update the weights with every iteration**. We first need to**shuffle the dataset so that we get a completely randomized dataset**. As the dataset is randomized and weights are updated for each single example, update of the weights and the cost function will be noisy jumping all over the place as shown below. Random sample helps to arrive at a global minima and avoids getting stuck at a local minima.

**Advantages of Stochastic Gradient Descent**

* Learning is much faster than batch gradient descent
* Redundancy is computation is removed as we take one training sample at a time for computation
* Weights can be updated on the fly for the new data samples as we take one training sample at a time for computation

**Disadvantages of Stochastic Gradient Descent**

* As we frequently update weights, Cost function fluctuates heavily

#### Mini Batch Gradient descent {#912f}

Mini-batch gradient is a variation of stochastic gradient descent where instead of single training example, mini-batch of samples is used.Mini batch gradient descent is widely used and converges faster and is more stable. Batch size can vary depending on the dataset. As we take a batch with different samples,it reduces the noise which is variance of the weight updates and that helps to have a more stable converge faster.

**Advantages of Min Batch Gradient Descent**

* Reduces variance of the parameter update and hence lead to stable convergence
* Speeds the learning
* Helpful to estimate the approximate location of the actual minimum

**Disadvantages of Mini Batch Gradient Descent**

* Loss is computed for each mini batch and hence total loss needs to be accumulated across all mini batches



