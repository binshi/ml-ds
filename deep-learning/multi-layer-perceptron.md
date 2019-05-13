Within DL, there are many different architectures: One such architecture is known as a convolutional neural net \(CNN\). Another architecture is known as a multi-layer perceptron, \(MLP\), etc. Different architectures lend themselves to solving different types of problems. An MLP is perhaps one of the most traditional types of DL architectures one may find, and that's when every element of a previous layer, is connected to every element of the next layer.

In MLPs, the matricies **W**i encode the transformation from one layer to another. \(Via a matrix multiply\). For example, if you have 10 neurons in one layer connected to 20 neurons of the next, then you will have a matrix **W**∈R10x20, that will map an input **v**∈R10x1 to an output **u**∈R1x20, via: **u**=**v**T**W**. Every column in **W**, encodes all the edges going from all the elements of a layer, to one of the elements of the next layer.

MLPs fell out of favor then, in part because they were hard to train. While there are many reasons for that hardship, one of them was also because their dense connections didn't allow them to scale easily for various computer vision problems. In other words, they did not have translation-equivariance baked in. This meant that if there was a signal in one part of the image that they needed to be sensitive to, they would need to re-learn how to be sensitive to it if that signal moved around. This wasted the capacity of the net, and so training became hard.This is where CNNs came in!

CNNs solved the signal-translation problem, because they would convolve each input signal with a detector, \(kernel\), and thus be sensitive to the same feature, but this time everywhere. In that case, our equation still looks the same, but the weight matricies **Wi**

are actually convolutional toeplitz matricies. The math is the same though.

