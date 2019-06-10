**One Hot Encoding**:

One variable for each class like in binary numbers where all numbers are represented by 1 or 0. If its one class then we can say its 0 or 1. But if its multiple classes then each class will have a variable.

eg:

[https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)

[https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)

[https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding](https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding)

[https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)

The main reason why we use sigmoid function is because it exists between **\(0 to 1\). **Therefore, it is especially used for models where we have to **predict the probability **as an output.Since probability of anything exists only between the range of **0 and 1, **sigmoid is the right choice.The function is **differentiable **.That means, we can find the slope of the sigmoid curve at any two points. The function is **monotonic **but function’s derivative is not. The logistic sigmoid function can cause a neural network to get stuck at the training time. The **softmax function **is a more generalized logistic activation function which is used for multiclass classification.

![](/assets/import.png)

**Regularization:**

In order to avoid overfitting with large weights \(sigmoid of larger numbers is always close to 1\) we start penalizing larger weight coefficients/weights by \(adding sum of the absolute value of weights or sum of weights squared\) multiplied by lambda to the error function.

L1: Small weights tend too go towards 0. Good for feature selection as it chooses only ones with maximum impact turning rest into 0. Sparsity \(1,0,1,0,0\)

L2: Tries to maintain  all weights homogenously small. Normally better for training models Sparsity \(0.5, 0.3, -0.2, 0.4, 0.1\)

**Dropout**: Randomly turn off some of the nodes in epochs so the other nodes have to pick up the slack and take more part in the training and thus ensures all nodes are contributing to the training

**Gradient Descent problems:**

**Local Minima**: While ‘searching’ for global minimum algorithm may encounter many ‘valleys’ whose bottoms we call local minimum. Depending on the type of algorithm being used, if the ‘valley’ is deep enough, the process might get stuck there and we end up with local minimum instead of global. One way to solve this is Random restarts and find gradient descents for all of them. Another way is to use momentum.  step = average of last few steps

![](/assets/Screenshot 2019-06-10 at 5.03.34 PM.png)

**Vanishing Gradient: **Certain activation functions, like the sigmoid function, squishes a large input space into a small input space between 0 and 1. Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Hence, the derivative becomes small. Gradients of neural networks are found using backpropagation. Simply put, backpropagation finds the derivatives of the network by moving layer by layer from the final layer to the initial one. By the chain rule, the derivatives of each layer are multiplied down the network \(from the final layer to the initial\) to compute the derivatives of the initial layers. However, when \_n \_hidden layers use an activation like the sigmoid function,\_n \_small derivatives are multiplied together. Thus, the gradient decreases exponentially as we propagate down to the initial layers. A small gradient means that the weights and biases of the initial layers will not be updated effectively with each training session. Since these initial layers are often crucial to recognizing the core elements of the input data, it can lead to overall inaccuracy of the whole network. The simplest solution is to use other activation functions, such as ReLU, which doesn’t cause a small derivative.

**Stochastic Gradient Descent and Batch Gradient Descent:**

The applicability of batch or stochastic gradient descent really depends on the error manifold expected.

Batch gradient descent computes the gradient using the whole dataset. This is great for convex, or relatively smooth error manifolds. In this case, we move somewhat directly towards an optimum solution, either local or global. Additionally, batch gradient descent, given an annealed learning rate, will eventually find the minimum located in it's basin of attraction.

Stochastic gradient descent \(SGD\) computes the gradient using a single sample. Most applications of SGD actually use a minibatch of several samples, for reasons that will be explained a bit later. SGD works well \(Not well, I suppose, but better than batch gradient descent\) for error manifolds that have lots of local maxima/minima. In this case, the somewhat noisier gradient calculated using the reduced number of samples tends to jerk the model out of local minima into a region that hopefully is more optimal. Single samples are really noisy, while minibatches tend to average a little of the noise out. Thus, the amount of jerk is reduced when using minibatches. A good balance is struck when the minibatch size is small enough to avoid some of the poor local minima, but large enough that it doesn't avoid the global minima or better-performing local minima. \(Incidently, this assumes that the best minima have a larger and deeper basin of attraction, and are therefore easier to fall into.\)

One benefit of SGD is that it's computationally a whole lot faster. Large datasets often can't be held in RAM, which makes vectorization much less efficient. Rather, each sample or batch of samples must be loaded, worked with, the results stored, and so on. Minibatch SGD, on the other hand, is usually intentionally made small enough to be computationally tractable.

Usually, this computational advantage is leveraged by performing many more iterations of SGD, making many more steps than conventional batch gradient descent. This usually results in a model that is very close to that which would be found via batch gradient descent, or better.

The way I like to think of how SGD works is to imagine that I have one point that represents my input distribution. My model is attempting to learn that input distribution. Surrounding the input distribution is a shaded area that represents the input distributions of all of the possible minibatches I could sample. It's usually a fair assumption that the minibatch input distributions are close in proximity to the true input distribution. Batch gradient descent, at all steps, takes the steepest route to reach the true input distribution. SGD, on the other hand, chooses a random point within the shaded area, and takes the steepest route towards this point. At each iteration, though, it chooses a new point. The average of all of these steps will approximate the true input distribution, usually quite well.

**If model is not working decrease the learning rate. **Decreasing learning rate. If steep: long steps, if plain small steps.

# Keras Optimizers {#keras-optimizers}

There are many optimizers in Keras, that we encourage you to explore further, in this[link](https://keras.io/optimizers/), or in this excellent[blog post](http://ruder.io/optimizing-gradient-descent/index.html#rmsprop). These optimizers use a combination of the tricks above, plus a few others. Some of the most common are:

#### SGD {#sgd}

This is Stochastic Gradient Descent. It uses the following parameters:

* Learning rate.
* Momentum \(This takes the weighted average of the previous steps, in order to get a bit of momentum and go over bumps, as a way to not get stuck in local minima\).
* Nesterov Momentum \(This slows down the gradient when it's close to the solution\).

#### Adam {#adam}

Adam \(Adaptive Moment Estimation\) uses a more complicated exponential decay that consists of not just considering the average \(first moment\), but also the variance \(second moment\) of the previous steps.

#### RMSProp {#rmsprop}

RMSProp \(RMS stands for Root Mean Squared Error\) decreases the learning rate by dividing it by an exponentially decaying average of squared gradients.

