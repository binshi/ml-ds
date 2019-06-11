[https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8](https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8)

[http://cs231n.stanford.edu/](http://cs231n.stanford.edu/)

[http://cs231n.github.io/](http://cs231n.github.io/)

[https://github.com/vdumoulin/conv\_arithmetic](https://github.com/vdumoulin/conv_arithmetic)

Checkpointing: [https://machinelearningmastery.com/check-point-deep-learning-models-keras/](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)

Grid search hyperparameters: [https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

[http://deeplearning.stanford.edu/wiki/index.php/Feature\_extraction\_using\_convolution](http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution)

**Pooling layers**: [https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)

**Image Augmentation**: [https://machinelearningmastery.com/image-augmentation-deep-learning-keras/](https://machinelearningmastery.com/image-augmentation-deep-learning-keras/)

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



