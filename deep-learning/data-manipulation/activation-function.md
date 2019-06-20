[https://www.learnopencv.com/understanding-activation-functions-in-deep-learning/](https://www.learnopencv.com/understanding-activation-functions-in-deep-learning/)

[https://keras.io/activations/](https://keras.io/activations/)

[https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/](https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/)

[https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c)

Activation functions are an extremely important feature of the artificial neural networks. They basically decide whether a neuron should be activated or not. Whether the information that the neuron is receiving is relevant for the given information or should it be ignored.

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/10/17123344/act.png)

The activation function is the non linear transformation that we do over the input signal. This transformed output is then sen to the next layer of neurons as input.

## Choosing the right Activation Function

Now that we have seen so many activation  functions, we need some logic / heuristics to know which activation function should be used in which situation. Good or bad – there is no rule of thumb.

However depending upon the properties of the problem we might be able to make a better choice for easy and quicker convergence of the network.

* Sigmoid functions and their combinations generally work better in the case of classifiers
* Sigmoids and tanh functions are sometimes avoided due to the vanishing gradient problem
* ReLU function is a general activation function and is used in most cases these days
* If we encounter a case of dead neurons in our networks the leaky ReLU function is the best choice
* Always keep in mind that ReLU function should only be used in the hidden layers
* As a rule of thumb, you can begin with using ReLU function and then move over to other activation functions in case ReLU doesn’t provide with optimum results

### Binary Step Function

The first thing that comes to our mind when we have an activation function would be a threshold based classifier i.e. whether or not the neuron should be activated. If the value Y is above a given threshold value then activate the neuron else leave it deactivated.

### Linear Function

We saw the problem with the step function, the gradient being zero, it was impossible to update gradient during the backpropagation. Instead of a simple step function, we can try using a linear function.

### Sigmoid

Sigmoid is a widely used activation function. It is of the form-

```
f(x)=1/(1+e^-x)
```

### Tanh

The tanh function is very similar to the sigmoid function. It is actually just a scaled version of the sigmoid function. Tanh works similar to the sigmoid function but is symmetric over the origin. it ranges from -1 to 1.

### ReLU

The ReLU function is the Rectified linear unit. It is the most widely used activation function. It is defined as-

```
f(x)=max(0,x)
```

ReLU is the most widely used activation function while designing networks today. First things first, the ReLU function is non linear, which means we can easily backpropagate the errors and have multiple layers of neurons being activated by the ReLU function.

The main advantage of using the ReLU function over other activation functions is that it does not activate all the neurons at the same time. What does this mean ? If you look at the ReLU function if the input is negative it will convert it to zero and the neuron does not get activated. This means that at a time only a few neurons are activated making the network sparse making it efficient and easy for computation. But ReLU also falls a prey to the gradients moving towards zero. If you look at the negative side of the graph, the gradient is zero, which means for activations in that region, the gradient is zero and the weights are not updated during back propagation. This can create dead neurons which never get activated. When we have a problem, we can always engineer a solution.

### Leaky ReLU

Leaky ReLU function is nothing but an improved version of the ReLU function. As we saw that for the ReLU function, the gradient is 0 for x&lt;0, which made the neurons die for activations in that region. Leaky ReLU is defined to address this problem. Instead of defining the Relu function as 0 for x less than 0, we define it as a small linear component of x.

### Softmax

The softmax function is also a type of sigmoid function but is handy when we are trying to handle classification problems. The sigmoid function as we saw earlier was able to handle just two classes. What shall we do when we are trying to handle multiple classes. Just classifying yes or no for a single class would not help then. The softmax function would squeeze the outputs for each class between 0 and 1 and would also divide by the sum of the outputs.

