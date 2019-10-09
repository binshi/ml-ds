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

**Transfer learning**: [https://www.hackerearth.com/practice/machine-learning/transfer-learning/transfer-learning-intro/tutorial/](https://www.hackerearth.com/practice/machine-learning/transfer-learning/transfer-learning-intro/tutorial/)

#### Categorical Cross Entropy\(To measure loss\)

![](/assets/Screenshot 2019-08-14 at 7.12.52 AM.png)

#### ![](/assets/Screenshot 2019-08-14 at 7.13.05 AM.png)**Model Validation**

![](/assets/Screenshot 2019-08-14 at 7.16.45 AM.png)In the above graph the intersection between validation and training is the best number of epochs. Lower the validation loss better the model.

**ModelCheckPoint: **

[https://keras.io/callbacks/\#modelcheckpoint](https://keras.io/callbacks/#modelcheckpoint)

[http://machinelearningmastery.com/check-point-deep-learning-models-keras/](http://machinelearningmastery.com/check-point-deep-learning-models-keras/)

**Checkpointing**

[https://machinelearningmastery.com/check-point-deep-learning-models-keras/](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)

When training deep learning models, the **checkpoint** is the weights of the model. These weights can be used to make predictions as is, or used as the basis for ongoing training. The Keras library provides a[checkpointing capability by a callback API](http://keras.io/callbacks/#modelcheckpoint).

The ModelCheckpoint callback class allows you to define where to checkpoint the model weights, how the file should named and under what circumstances to make a checkpoint of the model. The API allows you to specify which metric to monitor, such as loss or accuracy on the training or validation dataset. You can specify whether to look for an improvement in maximizing or minimizing the score. Finally, the filename that you use to store the weights can include variables like the[epoch number](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)or metric.

The ModelCheckpoint can then be passed to the training process when calling the fit\(\) function on the model.

![](/assets/Screenshot 2019-08-14 at 7.28.21 AM.png)

![](/assets/Screenshot 2019-08-14 at 7.31.08 AM.png)

![](/assets/Screenshot 2019-08-14 at 7.25.31 AM.png)

![](/assets/Screenshot 2019-08-14 at 7.31.50 AM.png)![](/assets/Screenshot 2019-08-14 at 7.33.07 AM.png)![](/assets/Screenshot 2019-08-14 at 7.33.50 AM.png)**CNN**

![](/assets/Screenshot 2019-08-14 at 7.39.25 AM.png)![](/assets/Screenshot 2019-08-14 at 7.39.48 AM.png)**Create you own filter**: [http://setosa.io/ev/image-kernels/](http://setosa.io/ev/image-kernels/)

![](/assets/Screenshot 2019-08-14 at 7.49.05 AM.png)For RGB images:

![](/assets/Screenshot 2019-08-14 at 7.51.19 AM.png)![](/assets/Screenshot 2019-08-14 at 8.00.27 AM.png)![](/assets/Screenshot 2019-08-14 at 8.01.06 AM.png)**Stride and Padding**

Stride the number of columns by which the filter is moved

Padding: Valid\(without padding\) and only those areas where the entire filter can be moved is used. In Same, 0 padding is used.![](/assets/stride.png)![](/assets/conv.png)

#### ![](/assets/Screenshot 2019-08-14 at 12.05.06 PM.png)![](/assets/Screenshot 2019-08-14 at 12.05.17 PM.png)![](/assets/Screenshot 2019-08-14 at 12.05.27 PM.png)Pooling Layers

**Pooling**

[https://keras.io/layers/pooling/](https://keras.io/layers/pooling/)

Neighboring pixels in images tend to have similar values, so conv layers will typically also produce similar values for neighboring pixels in outputs. As a result, **much of the information contained in a conv layer’s output is redundant**. For example, if we use an edge-detecting filter and find a strong edge at a certain location, chances are that we’ll also find relatively strong edges at locations 1 pixel shifted from the original one. However,**these are all the same edge! **We’re not finding anything new.

Pooling layers solve this problem. All they do is reduce the size of the input it’s given by \(you guessed it\)\_pooling\_values together in the input. The pooling is usually done by a simple operation like`max`,`min`, or`average`. To perform \_max \_pooling, we traverse the input image in 2x2 blocks \(because pool size = 2\) and put the \_max \_value into the output image at the corresponding pixel. **Pooling divides the input’s width and height by the pool size**. For our MNIST CNN, we’ll place a Max Pooling layer with a pool size of 2 right after our initial conv layer. The pooling layer will transform a 26x26x8 input into a 13x13x8 output

To increase the number of  nodes in a convolutional layer, increase number of filters. To increase the size of the detected  pattern increase the size of the filter

Always add a ReLU activation function to the`Conv2D`layers in your CNN. With the exception of the final layer in the network,`Dense`layers should also have a ReLU activation function.

When constructing a network for classification, the final layer in the network should be a`Dense`layer with a softmax activation function. The number of nodes in the final layer should equal the total number of classes in the dataset.

**Max Pooling Layer**![](/assets/Screenshot 2019-08-14 at 12.12.36 PM.png)

#### **Global Average Pooling layer**![](/assets/Screenshot 2019-08-14 at 12.13.39 PM.png)![](/assets/Screenshot 2019-08-14 at 12.21.23 PM.png)![](/assets/Screenshot 2019-08-14 at 12.22.36 PM.png)Pooling increases depth![](/assets/Screenshot 2019-08-14 at 12.22.40 PM.png)![](/assets/Screenshot 2019-08-14 at 12.26.21 PM.png)![](/assets/Screenshot 2019-08-14 at 12.27.41 PM.png)![](/assets/Screenshot 2019-08-14 at 1.22.09 PM.png)![](/assets/Screenshot 2019-08-14 at 1.22.18 PM.png)Image Augmentation![](/assets/Screenshot 2019-08-14 at 4.03.13 PM.png)

### CNN architecture Resources {#optional-resources}

* Check out the [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) paper!
* Read more about [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) here.
* The [ResNet](https://arxiv.org/pdf/1512.03385v1.pdf) paper can be found here.
* Here's the Keras [documentation](https://keras.io/applications/) for accessing some famous CNN architectures.
* Read this [detailed treatment](http://neuralnetworksanddeeplearning.com/chap5.html) of the vanishing gradients problem.
* Here's a GitHub [repository](https://github.com/jcjohnson/cnn-benchmarks) containing benchmarks for different CNN architectures.
* Visit the [ImageNet Large Scale Visual Recognition Competition \(ILSVRC\)](http://www.image-net.org/challenges/LSVRC/) website.

### CNN Visualization Resources {#-really-cool-optional-resources}

If you would like to know more about interpreting CNNs and convolutional layers in particular, you are encouraged to check out these resources:

* Here's a [section](http://cs231n.github.io/understanding-cnn/) from the Stanford's CS231n course on visualizing what CNNs learn.
* Check out this [demonstration](https://aiexperiments.withgoogle.com/what-neural-nets-see) of a cool [OpenFrameworks](http://openframeworks.cc/) app that visualizes CNNs in real-time, from user-supplied video!
* Here's a [demonstration](https://www.youtube.com/watch?v=AgkfIQ4IGaM&t=78s) of another visualization tool for CNNs. If you'd like to learn more about how these visualizations are made, check out this [video](https://www.youtube.com/watch?v=ghEmQSxT6tw&t=5s).
* Read this[Keras blog post](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)on visualizing how CNNs see the world. In this post, you can find an accessible introduction to Deep Dreams, along with code for writing your own deep dreams in Keras. When you've read that:

  * Also check out this [music video](https://www.youtube.com/watch?v=XatXy6ZhKZw) that makes use of Deep Dreams \(look at 3:15-3:40\)!
  * Create your own Deep Dreams \(without writing any code!\) using this [website](https://deepdreamgenerator.com/).

* If you'd like to read more about interpretability of CNNs:

  * Here's an [article](https://blog.openai.com/adversarial-example-research/) that details some dangers from using deep learning models \(that are not yet interpretable\) in real-world applications.
  * There's a lot of active research in this area. [These authors](https://arxiv.org/abs/1611.03530) recently made a step in the right direction.

# Visualizing CNNs {#visualizing-cnns}

Let’s look at an example CNN to see how it works in action.

The CNN we will look at is trained on ImageNet as described in[this paper](hhttps://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)by Zeiler and Fergus. In the images below \(from the same paper\), we’ll see\_what\_each layer in this network detects and see\_how\_each layer detects more and more complex ideas.

[![](https://video.udacity-data.com/topher/2017/April/58e91f1e_layer-1-grid/layer-1-grid.png)Example patterns that cause activations in the first layer of the network. These range from simple diagonal lines \(top left\) to green blobs \(bottom middle\).](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/cbf65dc4-c0b4-44c5-81c6-5997e409cb75#)

The images above are from Matthew Zeiler and Rob Fergus'[deep visualization toolbox](https://www.youtube.com/watch?v=ghEmQSxT6tw), which lets us visualize what each layer in a CNN focuses on.

Each image in the above grid represents a pattern that causes the neurons in the first layer to activate - in other words, they are patterns that the first layer recognizes. The top left image shows a -45 degree line, while the middle top square shows a +45 degree line. These squares are shown below again for reference.

[![](https://video.udacity-data.com/topher/2017/April/58e91f83_diagonal-line-1/diagonal-line-1.png)As visualized here, the first layer of the CNN can recognize -45 degree lines.](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/cbf65dc4-c0b4-44c5-81c6-5997e409cb75#)

[![](https://video.udacity-data.com/topher/2017/April/58e91f91_diagonal-line-2/diagonal-line-2.png)The first layer of the CNN is also able to recognize +45 degree lines, like the one above.](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/cbf65dc4-c0b4-44c5-81c6-5997e409cb75#)

Let's now see some example images that cause such activations. The below grid of images all activated the -45 degree line. Notice how they are all selected despite the fact that they have different colors, gradients, and patterns.

[![](https://video.udacity-data.com/topher/2017/April/58e91fd5_grid-layer-1/grid-layer-1.png)Example patches that activate the -45 degree line detector in the first layer.](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/cbf65dc4-c0b4-44c5-81c6-5997e409cb75#)

So, the first layer of our CNN clearly picks out very simple shapes and patterns like lines and blobs.

### Layer 2 {#layer-2}

[![](https://video.udacity-data.com/topher/2017/April/58e92033_screen-shot-2016-11-24-at-12.09.02-pm/screen-shot-2016-11-24-at-12.09.02-pm.png)A visualization of the second layer in the CNN. Notice how we are picking up more complex ideas like circles and stripes. The gray grid on the left represents how this layer of the CNN activates \(or "what it sees"\) based on the corresponding images from the grid on the right.](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/cbf65dc4-c0b4-44c5-81c6-5997e409cb75#)

The second layer of the CNN captures complex ideas.

As you see in the image above, the second layer of the CNN recognizes circles \(second row, second column\), stripes \(first row, second column\), and rectangles \(bottom right\).

**The CNN learns to do this on its own.**There is no special instruction for the CNN to focus on more complex objects in deeper layers. That's just how it normally works out when you feed training data into a CNN.

### Layer 3 {#layer-3}

[![](https://video.udacity-data.com/topher/2017/April/58e920b9_screen-shot-2016-11-24-at-12.09.24-pm/screen-shot-2016-11-24-at-12.09.24-pm.png)A visualization of the third layer in the CNN. The gray grid on the left represents how this layer of the CNN activates \(or "what it sees"\) based on the corresponding images from the grid on the right.](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/cbf65dc4-c0b4-44c5-81c6-5997e409cb75#)

The third layer picks out complex combinations of features from the second layer. These include things like grids, and honeycombs \(top left\), wheels \(second row, second column\), and even faces \(third row, third column\).

We'll skip layer 4, which continues this progression, and jump right to the fifth and final layer of this CNN.

### Layer 5 {#layer-5}

[![](https://video.udacity-data.com/topher/2017/April/58e9210c_screen-shot-2016-11-24-at-12.08.11-pm/screen-shot-2016-11-24-at-12.08.11-pm.png)A visualization of the fifth and final layer of the CNN. The gray grid on the left represents how this layer of the CNN activates \(or "what it sees"\) based on the corresponding images from the grid on the right.](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/cbf65dc4-c0b4-44c5-81c6-5997e409cb75#)

The last layer picks out the highest order ideas that we care about for classification, like dog faces, bird faces, and bicycles.

### Transfer Learning

# Transfer Learning {#transfer-learning}

Transfer learning involves taking a pre-trained neural network and adapting the neural network to a new, different data set.

Depending on both:

* the size of the new data set, and
* the similarity of the new data set to the original data set

the approach for using transfer learning will be different. There are four main cases:

1. new data set is small, new data is similar to original training data
2. new data set is small, new data is different from original training data
3. new data set is large, new data is similar to original training data
4. new data set is large, new data is different from original training data

[![](https://video.udacity-data.com/topher/2017/April/58e80aac_02-guide-how-transfer-learning-v3-01/02-guide-how-transfer-learning-v3-01.png)Four Cases when Using Transfer Learning](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

A large data set might have one million images. A small data could have two-thousand images. The dividing line between a large data set and small data set is somewhat subjective. Overfitting is a concern when using transfer learning with a small data set.

Images of dogs and images of wolves would be considered similar; the images would share common characteristics. A data set of flower images would be different from a data set of dog images.

Each of the four transfer learning cases has its own approach. In the following sections, we will look at each case one by one.

### Demonstration Network {#demonstration-network}

To explain how each situation works, we will start with a generic pre-trained convolutional neural network and explain how to adjust the network for each case. Our example network contains three convolutional layers and three fully connected layers:

[![](https://video.udacity-data.com/topher/2017/April/58e80ae2_02-guide-how-transfer-learning-v3-02/02-guide-how-transfer-learning-v3-02.png)General Overview of a Neural Network](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

Here is an generalized overview of what the convolutional neural network does:

* the first layer will detect edges in the image
* the second layer will detect shapes
* the third convolutional layer detects higher level features

Each transfer learning case will use the pre-trained convolutional neural network in a different way.

### Case 1: Small Data Set, Similar Data {#case-1-small-data-set-similar-data}

[![](https://video.udacity-data.com/topher/2017/April/58e80b0b_02-guide-how-transfer-learning-v3-03/02-guide-how-transfer-learning-v3-03.png)Case 1: Small Data Set with Similar Data](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

If the new data set is small and similar to the original training data:

* slice off the end of the neural network
* add a new fully connected layer that matches the number of classes in the new data set
* randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
* train the network to update the weights of the new fully connected layer

To avoid overfitting on the small data set, the weights of the original network will be held constant rather than re-training the weights.

Since the data sets are similar, images from each data set will have similar higher level features. Therefore most or all of the pre-trained neural network layers already contain relevant information about the new data set and should be kept.

Here's how to visualize this approach:

[![](https://video.udacity-data.com/topher/2017/April/58e80b31_02-guide-how-transfer-learning-v3-04/02-guide-how-transfer-learning-v3-04.png)Neural Network with Small Data Set, Similar Data](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

### Case 2: Small Data Set, Different Data {#case-2-small-data-set-different-data}

[![](https://video.udacity-data.com/topher/2017/April/58e80b55_02-guide-how-transfer-learning-v3-05/02-guide-how-transfer-learning-v3-05.png)Case 2: Small Data Set, Different Data](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

If the new data set is small and different from the original training data:

* slice off most of the pre-trained layers near the beginning of the network
* add to the remaining pre-trained layers a new fully connected layer that matches the number of classes in the new data set
* randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
* train the network to update the weights of the new fully connected layer

Because the data set is small, overfitting is still a concern. To combat overfitting, the weights of the original neural network will be held constant, like in the first case.

But the original training set and the new data set do not share higher level features. In this case, the new network will only use the layers containing lower level features.

Here is how to visualize this approach:

[![](https://video.udacity-data.com/topher/2017/April/58e80b82_02-guide-how-transfer-learning-v3-06/02-guide-how-transfer-learning-v3-06.png)Neural Network with Small Data Set, Different Data](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

### Case 3: Large Data Set, Similar Data {#case-3-large-data-set-similar-data}

[![](https://video.udacity-data.com/topher/2017/April/58e80ba3_02-guide-how-transfer-learning-v3-07/02-guide-how-transfer-learning-v3-07.png)Case 3: Large Data Set, Similar Data](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

If the new data set is large and similar to the original training data:

* remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
* randomly initialize the weights in the new fully connected layer
* initialize the rest of the weights using the pre-trained weights
* re-train the entire neural network

Overfitting is not as much of a concern when training on a large data set; therefore, you can re-train all of the weights.

Because the original training set and the new data set share higher level features, the entire neural network is used as well.

Here is how to visualize this approach:

[![](https://video.udacity-data.com/topher/2017/April/58e80bc3_02-guide-how-transfer-learning-v3-08/02-guide-how-transfer-learning-v3-08.png)Neural Network with Large Data Set, Similar Data](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

### Case 4: Large Data Set, Different Data {#case-4-large-data-set-different-data}

[![](https://video.udacity-data.com/topher/2017/April/58e80bf7_02-guide-how-transfer-learning-v3-09/02-guide-how-transfer-learning-v3-09.png)Case 4: Large Data Set, Different Data](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

If the new data set is large and different from the original training data:

* remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
* retrain the network from scratch with randomly initialized weights
* alternatively, you could just use the same strategy as the "large and similar" data case

Even though the data set is different from the training data, initializing the weights from the pre-trained network might make training faster. So this case is exactly the same as the case with a large, similar data set.

If using the pre-trained network as a starting point does not produce a successful model, another option is to randomly initialize the convolutional neural network weights and train the network from scratch.

Here is how to visualize this approach:

[![](https://video.udacity-data.com/topher/2017/April/58e80c1c_02-guide-how-transfer-learning-v3-10/02-guide-how-transfer-learning-v3-10.png)Neural Network with Large Data Set, Different Data](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/921922f7-359e-4998-9e56-d7336861f8ae/lessons/52fc79a7-13ff-4065-b3c6-8203ec9ef60c/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

### Optional Resources {#optional-resources}

* Check out this [research paper](https://arxiv.org/pdf/1411.1792.pdf)  that systematically analyzes the transferability of features learned in pre-trained CNNs.
* Read the
  [Nature publication](http://www.nature.com/articles/nature21056.epdf?referrer_access_token=_snzJ5POVSgpHutcNN4lEtRgN0jAjWel9jnR3ZoTv0NXpMHRAJy8Qn10ys2O4tuP9jVts1q2g1KBbk3Pd3AelZ36FalmvJLxw1ypYW0UxU7iShiMp86DmQ5Sh3wOBhXDm9idRXzicpVoBBhnUsXHzVUdYCPiVV0Slqf-Q25Ntb1SX_HAv3aFVSRgPbogozIHYQE3zSkyIghcAppAjrIkw1HtSwMvZ1PXrt6fVYXt-dvwXKEtdCN8qEHg0vbfl4_m&tracking_referrer=edition.cnn.com)
  detailing Sebastian Thrun's cancer-detecting CNN!



