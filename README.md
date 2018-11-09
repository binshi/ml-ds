# My Awesome Book

This file file serves as your book's preface, a great place to describe your book's content and ideas.



To have a basic mathematical background, you need to have some knowledge of the following mathematical concepts:  
- Probability and statistics  
- Linear algebra  
- Optimization  
- Multivariable calculus  
- Functional analysis \(not essential\)  
- First-order logic \(not essential\)  
You can find some reasonable material on most of these by searching for "&lt;topic&gt; lecture notes" on Google. Usually, you'll find good lecture notes compiled by some professor teaching that course. The first few results should give you a good set to choose from. See[Prasoon Goyal's answer to How should I start learning the maths for machine learning and from where?](https://www.quora.com/How-should-I-start-learning-the-maths-for-machine-learning-and-from-where/answer/Prasoon-Goyal)  
Skim through these. You don't need to go through them in a lot of detail. You can come back to studying the math as and when required while learning ML.

Then, for a quick overview of ML, you can follow the roadmap below \(I’ve written about many of these topics in various answers on Quora; I linked the most relevant ones for quick reference.\)

**Day 1:**

* Basic terminology:

1. 1. Most common settings: Supervised setting, Unsupervised setting, Semi-supervised setting, Reinforcement learning.
   2. Most common problems: Classification \(binary 
      &
       multiclass\), Regression, Clustering.
   3. Preprocessing of data: Data normalization.

* Concepts of hypothesis sets, empirical error, true error,
  [complexity of hypotheses sets](https://www.quora.com/What-is-hyperparameter-optimization-in-machine-learning-in-formal-terms/answer/Prasoon-Goyal), [regularizatin](https://www.quora.com/What-is-an-intuitive-explanation-of-regularization/answer/Prasoon-Goyal), [bias-variance trade-off](https://www.quora.com/Is-there-any-theorem-in-statistics-or-machine-learning-that-shows-that-the-bigger-the-dataset-the-bigger-the-accuracy/answer/Prasoon-Goyal), [loss functions](https://www.quora.com/MSE-is-a-convex-loss-func-for-linear-and-logistic-regression-how-come-it-isnt-for-NN-and-backprop-i-e-local-optimums/answer/Prasoon-Goyal), [cross-validation](https://www.quora.com/Should-I-split-my-data-to-train-test-split-or-train-validation-test-subset/answer/Prasoon-Goyal)
  .

**Day 2:**

* [Optimization](https://www.quora.com/How-much-of-machine-learning-is-actually-just-optimization/answer/Prasoon-Goyal)
  basics:

1. 1. Terminology & Basic concepts: Convex optimization, Lagrangian, Primal-dual problems, Gradients & subgradients,
      ℓ1 and ℓ2 [regularized objective functions](https://www.quora.com/Why-small-l1-norm-means-sparsity/answer/Prasoon-Goyal)
   2. Algorithms:
      [Batch gradient descent & stochastic gradient descent](https://www.quora.com/What-should-everybody-know-about-stochastic-gradient-descent/answer/Prasoon-Goyal)
      , Coordinate gradient descent.
   3. Implementation: Write code for stochastic gradient descent for a simple objective function, tune the step size, and get an intuition of the algorithm.

**Day 3:**

* Classification:

1. 1. Logistic Regression
   2. [Support vector machines](https://www.quora.com/What-are-the-tricky-ideas-introduced-in-Support-Vector-Machines-SVM-And-what-makes-them-successful/answer/Prasoon-Goyal): Geometric intuition, primal-dual formulations, notion of support vectors, kernel trick, understanding of [hyperparameters](https://www.quora.com/What-is-hyperparameter-optimization-in-machine-learning-in-formal-terms/answer/Prasoon-Goyal), [grid search](https://www.quora.com/Machine-Learning-How-does-grid-search-work/answer/Prasoon-Goyal)
   3. [Online tool for SVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
      : Play with this online SVM tool \(scroll down to “Graphic Interface”\) to get some intuition of the algorithm.

**Day 4:**

* Regression:

1. 1. Ridge regression

* Clustering:

1. 1. k-means & Expectation-Maximization algorithm.
   2. Top-down and bottom-up hierarchical clustering.

**Day 5:**

* Bayesian methods:

1. 1. Basic terminology: Priors, posteriors, likelihood, maximum likelihood estimation and maximum-a-posteriori inference.
   2. [Gaussian Mixture Models](https://www.quora.com/What-is-a-generative-model/answer/Prasoon-Goyal)
   3. [Latent Dirichlet Allocation](https://www.quora.com/What-are-the-best-machine-learning-algorithms-to-determine-the-topic-or-theme-of-a-file-content/answer/Prasoon-Goyal)
      : The generative model and basic idea of parameter estimation.

**Day 6:**

* Graphical models:

1. 1. Basic terminology: Bayesian networks, Markov networks / Markov random fields.
   2. Inference algorithms: Variable elimination, Belief propagation.
   3. Simple examples: Hidden Markov Models. Ising model.

**Days 7–8:**

* [Neural Networks](https://www.quora.com/What-is-a-deep-learning-algoritm-simply-explained/answer/Prasoon-Goyal)

1. 1. Basic terminology: Neuron, Activation function, Hidden layer.
   2. Convolutional neural networks: Convolutional layer, pooling layer, Backpropagation.
   3. Memory-based neural networks:
      [Recurrent Neural Networks](https://www.quora.com/How-are-recurrent-neural-networks-different-from-convolutional-neural-networks/answer/Prasoon-Goyal)
      , Long-short term memory.
   4. Tutorials: I’m familiar with
      [this Torch tutorial](https://github.com/clementfarabet/ipam-tutorials/tree/master/th_tutorials)
      \(you’ll want to look at
      𝟷\_𝚜𝚞𝚙𝚎𝚛𝚟𝚒𝚜𝚎𝚍
      1\_supervised
      directory\). There might be other tutorials in other deep learning frameworks.

**Day 9:**

* Miscellaneous topics:

1. 1. Decision trees
   2. Recommender systems
   3. Markov decision processes
   4. Multi-armed bandits

**Day 10:**\(Budget day\)

* You can use the last day to catch up on anything left from previous days, or learn more about whatever topic you found most interesting / useful for your future work.

---

Once you’ve gone through the above, you’ll want to start going through some standard online course or ML text. Andrew Ng's course on Coursera is a good starting point. An advanced version of the course is available on The Open Academy \([Machine Learning \| The Open Academy](http://theopenacademy.com/content/machine-learning)\). The popular books that I have some experience with are the following:

* [Pattern Recognition and Machine Learning: Christopher Bishop](http://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738/ref=sr_1_1?ie=UTF8&keywords=bishop+machine+learning&qid=1436657115&sr=8-1)
* [Machine Learning: A Probabilistic Perspective: Kevin P. Murphy](http://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020/ref=sr_1_2?ie=UTF8&keywords=bishop+machine+learning&qid=1436657115&sr=8-2)

While Murphy's book is more current and is more elaborate, I find Bishop’s to be more accessible for beginners. You can choose one of them according to your level.  
At this point, you should have a working knowledge of machine learning. Beyond this, if you're interested in a particular topic, look for specific online resources on the topic, read seminal papers in the subfield, try finding some simpler problems and implement them.

It is important to implement some basic algorithms when you start doing ML, like gradient descent, AdaBoost, decision trees, etc \[in addition to whatever you’ve implemented in the 10-day overview\]. You should also have some experience with data preprocessing, normalization, etc. Once you have implemented a few algorithms from scratch, for other algorithms, you should use the standard implementations \(like LibSVM, Weka, ScikitLearn, etc\) on some toy problems, and get a good understanding of different algorithms.

