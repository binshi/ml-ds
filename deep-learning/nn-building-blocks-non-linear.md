## Neural Network Building Blocks - Non Linear![](/assets/Screenshot 2019-08-12 at 11.09.41 AM.png)![](/assets/Screenshot 2019-08-12 at 11.11.02 AM.png)![](/assets/Screenshot 2019-08-12 at 11.11.52 AM.png)![](/assets/Screenshot 2019-08-12 at 11.13.42 AM.png)

#### Feedforward

Feedforward is the process neural networks use to turn the input into an output.

#### ![](/assets/Screenshot 2019-08-12 at 11.19.12 AM.png)![](/assets/Screenshot 2019-08-12 at 11.19.49 AM.png)Error function

![](/assets/Screenshot 2019-08-12 at 11.20.55 AM.png)![](/assets/Screenshot 2019-08-12 at 11.21.40 AM.png)

#### Backpropagation

Now, we're ready to get our hands into training a neural network. For this, we'll use the method known as**backpropagation**. In a nutshell, backpropagation will consist of:

* Doing a feedforward operation.
* Comparing the output of the model with the desired output.
* Calculating the error.
* Running the feedforward operation backwards \(backpropagation\) to spread the error to each of the weights.
* Use this to update the weights, and get a better model.
* Continue this until we have a model that is good.

Try to call the boundary towards the misclassified point and update the weights to w1' and w2'![](/assets/Screenshot 2019-08-12 at 11.54.11 AM.png)![](/assets/Screenshot 2019-08-12 at 11.54.48 AM.png)**Multi-Layer Perceptrons**

![](/assets/Screenshot 2019-08-12 at 11.57.43 AM.png)![](/assets/Screenshot 2019-08-12 at 12.00.33 PM.png)![](/assets/Screenshot 2019-08-12 at 12.02.02 PM.png)![](/assets/Screenshot 2019-08-12 at 12.02.31 PM.png)![](/assets/Screenshot 2019-08-12 at 12.02.58 PM.png)![](/assets/Screenshot 2019-08-12 at 12.07.56 PM.png)![](/assets/Screenshot 2019-08-12 at 12.10.21 PM.png)![](/assets/Screenshot 2019-08-12 at 12.11.35 PM.png)![](/assets/Screenshot 2019-08-12 at 12.18.08 PM.png)![](/assets/Screenshot 2019-08-12 at 12.18.31 PM.png)![](/assets/Screenshot 2019-08-12 at 5.09.22 PM.png)

#### Training Optimization

##### Model Complexity Graph

![](/assets/Screenshot 2019-08-12 at 6.19.47 PM.png)![](/assets/Screenshot 2019-08-12 at 6.21.04 PM.png)

**Subtle overfitting**

![](/assets/Screenshot 2019-08-12 at 6.23.31 PM.png)![](/assets/Screenshot 2019-08-12 at 6.26.37 PM.png)**Regularization to punish large coefficients**![](/assets/Screenshot 2019-08-12 at 6.28.26 PM.png)![](/assets/Screenshot 2019-08-12 at 6.31.43 PM.png)**Dropout**

Dropout is a regularization method that approximates training a large number of neural networks with different architectures in parallel. During training, some number of layer outputs are randomly ignored or “_dropped out_.” This has the effect of making the layer look-like and be treated-like a layer with a different number of nodes and connectivity to the prior layer. In effect, each update to a layer during training is performed with a different “_view_” of the configured layer.

* Dropout forces a neural network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.
* Dropout roughly doubles the number of iterations required to converge. However, training time for each epoch is less.
* With H hidden units, each of which can be dropped, we have 2^H possible models. In testing phase, the entire network is considered and each activation is reduced by a factor _p._

**Local Minima**![](/assets/Screenshot 2019-08-12 at 6.37.02 PM.png)

**Vanishing Gradient**

If we calculate the derivative of the point at the extreme left oor right it is almost 0. Derivative is what tells us in which direction to move.

![](/assets/Screenshot 2019-08-12 at 6.38.05 PM.png)![](/assets/Screenshot 2019-08-12 at 6.39.27 PM.png)

**Resolution to solve vanishing gradient**![](/assets/Screenshot 2019-08-12 at 6.40.07 PM.png)![](/assets/Screenshot 2019-08-12 at 6.40.35 PM.png)![](/assets/Screenshot 2019-08-12 at 6.41.33 PM.png)

**Learning Rate**![](/assets/Screenshot 2019-08-12 at 6.50.10 PM.png)To avoid local minima![](/assets/Screenshot 2019-08-12 at 6.50.45 PM.png)![](/assets/Screenshot 2019-08-12 at 6.52.13 PM.png)

# Keras Optimizers {#keras-optimizers}

There are many optimizers in Keras, that we encourage you to explore further, in this [link](https://keras.io/optimizers/), or in this excellent [blog post](http://ruder.io/optimizing-gradient-descent/index.html#rmsprop). These optimizers use a combination of the tricks above, plus a few others. Some of the most common are:

#### SGD {#sgd}

This is Stochastic Gradient Descent. It uses the following parameters:

* Learning rate.
* Momentum \(This takes the weighted average of the previous steps, in order to get a bit of momentum and go over bumps, as a way to not get stuck in local minima\).
* Nesterov Momentum \(This slows down the gradient when it's close to the solution\).

#### Adam {#adam}

Adam \(Adaptive Moment Estimation\) uses a more complicated exponential decay that consists of not just considering the average \(first moment\), but also the variance \(second moment\) of the previous steps.

#### RMSProp {#rmsprop}

RMSProp \(RMS stands for Root Mean Squared Error\) decreases the learning rate by dividing it by an exponentially decaying average of squared gradients.

