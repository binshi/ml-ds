[https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8](https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8)

[http://cs231n.stanford.edu/](http://cs231n.stanford.edu/)

[http://cs231n.github.io/](http://cs231n.github.io/)

[https://github.com/vdumoulin/conv\_arithmetic](https://github.com/vdumoulin/conv_arithmetic)

Tensorflow CNN:** **[https://www.tensorflow.org/tutorials/estimators/cnn](https://www.tensorflow.org/tutorials/estimators/cnn)

**Checkpointing**: [https://machinelearningmastery.com/check-point-deep-learning-models-keras/](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)

**Grid search hyperparameters**: [https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

[http://deeplearning.stanford.edu/wiki/index.php/Feature\_extraction\_using\_convolution](http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution)

**Pooling layers**: [https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)

**Image Augmentation**: [https://machinelearningmastery.com/image-augmentation-deep-learning-keras/](https://machinelearningmastery.com/image-augmentation-deep-learning-keras/)

[https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

If you would like to know more about interpreting CNNs and convolutional layers in particular, you are encouraged to check out these resources:

* Here's a[section](http://cs231n.github.io/understanding-cnn/)from the Stanford's CS231n course on visualizing what CNNs learn.

* Check out this[demonstration](https://aiexperiments.withgoogle.com/what-neural-nets-see)of a cool[OpenFrameworks](http://openframeworks.cc/)app that visualizes CNNs in real-time, from user-supplied video!

* Here's a[demonstration](https://www.youtube.com/watch?v=AgkfIQ4IGaM&t=78s)of another visualization tool for CNNs. If you'd like to learn more about how these visualizations are made, check out this[video](https://www.youtube.com/watch?v=ghEmQSxT6tw&t=5s).

* Read this[Keras blog post](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)on visualizing how CNNs see the world. In this post, you can find an accessible introduction to Deep Dreams, along with code for writing your own deep dreams in Keras. When you've read that:

  * Also check out this [music video](https://www.youtube.com/watch?v=XatXy6ZhKZw) that makes use of Deep Dreams \(look at 3:15-3:40\)!
  * Create your own Deep Dreams \(without writing any code!\) using this [website](https://deepdreamgenerator.com/)

* If you'd like to read more about interpretability of CNNs:

  * Here's an [article](https://blog.openai.com/adversarial-example-research/) that details some dangers from using deep learning models \(that are not yet interpretable\) in real-world applications.
  * There's a lot of active research in this area.[These authors](https://arxiv.org/abs/1611.03530) recently made a step in the right direction.

If you'd like more details about fully connected layers in Keras, check out the [documentation](https://keras.io/layers/core/) for the Dense layer. You can change the way the weights are initialized through supplying values for the `kernel_initializer`and `bias_initializer`parameters. Note that the default values are `'glorot_uniform'`, and `'zeros'`, respectively. You can read more about how each of these initializers work in the corresponding Keras [documentation](https://keras.io/initializers/).

There are many different [loss functions](https://keras.io/losses/) in Keras. For this lesson, we will only use`categorical_crossentropy`

Check out the [list of available optimizers](https://keras.io/optimizers/) in Keras. The optimizer is specified when you compile the model \(in Step 7 of the notebook\).

* `'sgd'`: SGD
* `'rmsprop'`: RMSprop
* `'adagrad'`Adagrad
* `'adadelta'`: Adadelta
* `'adam'`: Adam
* `'adamax'`: Adamax
* `'nadam'`: Nadam
* `'tfoptimizer'`: TFOptimizer

* There are many callbacks \(such as ModelCheckpoint\) that you can use to monitor your model during the training process. If you'd like, check out the [**details**](https://keras.io/callbacks/#modelcheckpoint) here. You're encouraged to begin with learning more about the EarlyStopping callback. If you'd like to see another code example of ModelCheckpoint, check out  [**this blog**](http://machinelearningmastery.com/check-point-deep-learning-models-keras/)  
  .

**Checkpointing**

[https://machinelearningmastery.com/check-point-deep-learning-models-keras/](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)

When training deep learning models, the **checkpoint** is the weights of the model. These weights can be used to make predictions as is, or used as the basis for ongoing training. The Keras library provides a[checkpointing capability by a callback API](http://keras.io/callbacks/#modelcheckpoint).

The ModelCheckpoint callback class allows you to define where to checkpoint the model weights, how the file should named and under what circumstances to make a checkpoint of the model. The API allows you to specify which metric to monitor, such as loss or accuracy on the training or validation dataset. You can specify whether to look for an improvement in maximizing or minimizing the score. Finally, the filename that you use to store the weights can include variables like the[epoch number](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)or metric.

The ModelCheckpoint can then be passed to the training process when calling the fit\(\) function on the model.

**Pooling**

Neighboring pixels in images tend to have similar values, so conv layers will typically also produce similar values for neighboring pixels in outputs. As a result,**much of the information contained in a conv layer’s output is redundant**. For example, if we use an edge-detecting filter and find a strong edge at a certain location, chances are that we’ll also find relatively strong edges at locations 1 pixel shifted from the original one. However,**these are all the same edge!**We’re not finding anything new.

Pooling layers solve this problem. All they do is reduce the size of the input it’s given by \(you guessed it\)\_pooling\_values together in the input. The pooling is usually done by a simple operation like`max`,`min`, or`average`. To perform \_max \_pooling, we traverse the input image in 2x2 blocks \(because pool size = 2\) and put the \_max \_value into the output image at the corresponding pixel. **Pooling divides the input’s width and height by the pool size**. For our MNIST CNN, we’ll place a Max Pooling layer with a pool size of 2 right after our initial conv layer. The pooling layer will transform a 26x26x8 input into a 13x13x8 output

To increase the number of  nodes in a convolutional layer, increase number of filters. To increase the size of the detected  pattern increase the size of the filter

Always add a ReLU activation function to the`Conv2D`layers in your CNN. With the exception of the final layer in the network,`Dense`layers should also have a ReLU activation function.

When constructing a network for classification, the final layer in the network should be a`Dense`layer with a softmax activation function. The number of nodes in the final layer should equal the total number of classes in the dataset.

**Transfer learning**: [https://www.hackerearth.com/practice/machine-learning/transfer-learning/transfer-learning-intro/tutorial/](https://www.hackerearth.com/practice/machine-learning/transfer-learning/transfer-learning-intro/tutorial/)

#### Categorical Cross Entropy\(To measure loss\)

![](/assets/Screenshot 2019-08-14 at 7.12.52 AM.png)

#### ![](/assets/Screenshot 2019-08-14 at 7.13.05 AM.png)**Model Validation**

![](/assets/Screenshot 2019-08-14 at 7.16.45 AM.png)In the above graph the intersection between validation and training is the best number of epochs. Lower the validation loss better the model.

**ModelCheckPoint: **

[https://keras.io/callbacks/\#modelcheckpoint](https://keras.io/callbacks/#modelcheckpoint)

[http://machinelearningmastery.com/check-point-deep-learning-models-keras/](http://machinelearningmastery.com/check-point-deep-learning-models-keras/)

![](/assets/Screenshot 2019-08-14 at 7.28.21 AM.png)

![](/assets/Screenshot 2019-08-14 at 7.31.08 AM.png)

![](/assets/Screenshot 2019-08-14 at 7.25.31 AM.png)

![](/assets/Screenshot 2019-08-14 at 7.31.50 AM.png)![](/assets/Screenshot 2019-08-14 at 7.33.07 AM.png)![](/assets/Screenshot 2019-08-14 at 7.33.50 AM.png)

