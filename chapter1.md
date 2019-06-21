# Appendix:

[https://developers.google.com/machine-learning/glossary/](https://developers.google.com/machine-learning/glossary/)

**Regression**: a measure of the relation between the mean value of one variable \(e.g. output\) and corresponding values of other variables \(e.g. time and cost\).

**probability**: A measure of uncertainty which lies between 0 and 1, where 0 means impossible and 1 means certain. Probabilities are often expressed as a percentages \(such as 0%, 50% and 100%\).

**random variable**: A variable \(a named quantity\) whose value is uncertain.

**normalization constraint**:The constraint that the[probabilities](http://mbmlbook.com/MurderMystery.html#probability)given by a[probability distribution](http://mbmlbook.com/MurderMystery.html#probability_distribution)must add up to 1 over all possible values of the[random variable](http://mbmlbook.com/MurderMystery.html#random_variable). For example, for aBernoulli\(p\)Bernoulli\(p\)distribution the[probability](http://mbmlbook.com/MurderMystery.html#probability)oftrueis ppand so the[probability](http://mbmlbook.com/MurderMystery.html#probability)of the only other statefalsemust be1−p1−p.

**probability distribution**: A function which gives the[probability](http://mbmlbook.com/MurderMystery.html#probability)for every possible value of a[random variable](http://mbmlbook.com/MurderMystery.html#random_variable). Written asP\(A\)P\(A\)for a[random variable](http://mbmlbook.com/MurderMystery.html#random_variable)A.

**sampling**: Randomly choosing a value such that the[probability](http://mbmlbook.com/MurderMystery.html#probability)of picking any particular value is given by a[probability distribution](http://mbmlbook.com/MurderMystery.html#probability_distribution). This is known as sampling from the distribution. For example, here are 10 samples from aBernoulli\(0.7\) distribution:false,true,false,false,true,true,true,false,trueandtrue. If we took a very large number of samples from aBernoulli\(0.7\) distribution then the percentage of the samples equal totruewould be very close to 70%.

**Bernoulli distribution**: A[probability distribution](http://mbmlbook.com/MurderMystery.html#probability_distribution)over a two-valued \(binary\)[random variable](http://mbmlbook.com/MurderMystery.html#random_variable). The Bernoulli distribution has one parameterppwhich is the[probability](http://mbmlbook.com/MurderMystery.html#probability)of the valuetrueand is written asBernoulli\(p\)\(p\). As an example,Bernoulli\(0.5\)\(0.5\)represents the uncertainty in the outcome of a fair coin toss.

**uniform distribution**: A[probability distribution](http://mbmlbook.com/MurderMystery.html#probability_distribution)where every possible value is equally probable. For example,Bernoulli\(0.5\)\(0.5\)is a uniform distribution sincetrueandfalseboth have the same[probability](http://mbmlbook.com/MurderMystery.html#probability)\(of 0.5\) and these are the only possible values.

**point mass**: A distribution which gives[probability](http://mbmlbook.com/MurderMystery.html#probability)1 to one value and[probability](http://mbmlbook.com/MurderMystery.html#probability)0 to all other values, which means that the[random variable](http://mbmlbook.com/MurderMystery.html#random_variable)is certain to have the specified value. For example,Bernoulli\(1\)Bernoulli\(1\)is a point mass indicating that the variable is certain to betrue.

**Cost Function: **The cost function compares how the hypothesis matches to the dataset and gives it a number, which represents the error. Thus, we want to find the hypothesis that minimizes the cost function or has the lowest error.

**Flattening: **the fully connected layer expects a **vector **as input. Convolution outputs a series of filters, which each are a grid shape. Flattening specifies a function mapping from these filters to a vector, so you can backpropagate errors back through the convolutional layers.

**Checkpointing: **[Application checkpointing](https://en.wikipedia.org/wiki/Application_checkpointing) is a fault tolerance technique for long running processes. It is an approach where a snapshot of the state of the system is taken in case of system failure. If there is a problem, not all is lost. The checkpoint may be used directly, or used as the starting point for a new run, picking up where it left off.

A vector is a series of numbers. It is like a matrix with only one row but multiple columns \(or only one column but multiple rows\). An example is: \[1,2,3,5,6,3,2,0\].

**vector**: A feature vector is just a vector that contains information describing an object's important characteristics.

**Feature  vector**: In image processing, features can take many forms. A simple feature representation of an image is the raw intensity value of each pixel. However, more complicated feature representations are also possible. For facial expression analysis, I use mostly SIFT descriptor features \(scale invariant feature transform\). These features capture the prevalence of different line orientations.

**Color spaces**:

**RGB** is unintuitive when defining exact shades of a color if you need to define a particular \_range\_of colors \(useful when tracking objects in a video stream based on color appearance\).

The **HSV** color space tends to be more intuitive in terms of actually defining a particular color \(or range\), but it doesn’t do a great job of representing how humans see and interpret color.

Then we have the **L\*a\*b\*** color space — this color space tries to mimic the methodology in which humans see and interpret color. This implies that the[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between two arbitrary colors in the L\*a\*b\* color space have actual \_perceptual meaning. \_The addition of the perceptual meaning property makes the L\*a\*b\* color space less intuitive and easy to understand than RGB or HSV, but because of the perceptual meaning property, we often use it in computer vision.

**ROC Curve: **

The **confusion matrix** is a table that summarizes how successful the classification model is at predicting examples belonging to various classes. One axis of the confusion matrix is the label that the model predicted, and the other axis is the actual label.

To create a **regularized** model, we modify the objective function by adding a penalizing term whose value is higher when the model is more complex.

**Sensitivity**: Of all the people **with** cancer, how many were correctly diagnosed?

**Specificity**: Of all the people **without** cancer, how many were correctly diagnosed?

**Recall**: Of all the people who **have cancer**, how many did **we diagnose** as having cancer?

**Precision**: Of all the people **we diagnosed** with cancer, how many actually **had cancer**?

**Ensemble learning **is the process by which multiple models, such as classifiers or experts, are strategically generated and combined to solve a particular computational intelligence problem.

Overfitting is a serious problem in networks Deep neural nets with a large number of parameters. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. **Dropout** is a technique for addressing this problem. The key idea is to randomly drop units \(along with their connections\) from the neural network during training. This prevents units from co-adapting too much. During training, dropout samples from an exponential number of different “thinned” networks. At test time, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights. This significantly reduces overfitting and gives major improvements over other regularization methods.

Vanishing Gradient:

The problem of transforming raw data into a dataset is called **feature engineering**

The **error function** needs to be differentiable. Hence it needs to be continuous.

Log Loss Error function:

Activation function It’s just a thing function that you use to get the output of node. It is also known as **Transfer Function**

**Sigmoid** to convert number between 0 and 1. Add weights subtract bias and then sigmoid it

Model Complexity Graph:

![](/assets/Screenshot 2019-06-10 at 4.17.30 PM.png)

One can identify the optimal number of **epochs** from the graph drawn between epochs and the training-validation loss or graph drawn between epochs and training-validation accuracy

**MLP\(Multi Layer  Perceptons\)** uses fully connected layers and only accepts vectors\(from matrices using flatten\(\)\) as inputs.On the other hand CNNs can also  use sparsely connected layers and also accept matrices as inputs

[https://keras.io/callbacks/](https://keras.io/callbacks/) EarlyStopping

