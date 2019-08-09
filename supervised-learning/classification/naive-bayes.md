## Naive bayes

Source: https://towardsdatascience.com/naive-bayes-intuition-and-implementation-ac328f9c9718

In a broad sense, Naive Bayes models are a special kind of classification machine learning algorithms. They are based on a statistical classification technique called ‘Bayes Theorem’.

Naive Bayes model are called ‘naive’ algorithms becaused they make an assumption that the predictor variables are independent from each other. In other words, that the presence of a certain feature ina dataset is completely unrelated to the presence of any other feature.

They provide aneasy way to build accurate models with very good performance given their simplicity.

They do this by providing a way to calculate the ‘posterior’ probability of a certain event _A_ to occur_,_ given some probabilities of ‘prior’ events.

![](https://miro.medium.com/max/60/1*Lt8E08oxEnnRegLbNBzNAg.png?q=20)

![](https://miro.medium.com/max/700/1*Lt8E08oxEnnRegLbNBzNAg.png)

# Example {#3716}

We will introduce the main concepts regarding Navive Bayes algorithm, by studying an example:

Let’s consider the case of two colleagues that work in the same office: Alice and Bruno. And we know that:

* Alice comes to the office 3 days a week.
* Bruno comes to the office 1days a week.

This will be our ‘prior’ information.

We are at the office and we see passing across us someone very fast, so fast that we don’t know who the person is: Alice or Bruno.

Given the information that we have until know and assuming that they only work 4 days a week, the probabilities of the person seen to be either Alice or Bruno are:

* P\(Alice\) = 3/4 = 0.75
* P\(Bruno\) = 1/4 = 0.25

When we saw the person passing by, we saw that he/she was wearing a red jacket. We also know the following:

* Alice wears red 2 times a week.
* Bruno wears red 3 times a week.

So for every workweek, that has 5 days, we can infer the following:

* The probability of Alice to wear red is → P\(Red\|Alice\) = 2/5 = 0.4
* The probability of Bruno to wear red is → P\(Red\|Bruno\) = 3/5 = 0.6

This new probabilities will be the ‘posterior’ information.

So, with this information, who did we see passing by?

![](https://miro.medium.com/max/60/1*ww34WC9G-rVlOvZZ2-YCoQ.png?q=20)

![](https://miro.medium.com/max/700/1*ww34WC9G-rVlOvZZ2-YCoQ.png)

Initially, we knew the probability of P\(Alice\) and P\(Bruno\), and later we infered the probabilties of P\(Red\|Alice\) and P\(Red\|Bruno\).

So, the real probabilities are:

![](https://miro.medium.com/max/60/1*KJmnV8T6e2kgH73VKkIPVA.png?q=20)

![](https://miro.medium.com/max/700/1*KJmnV8T6e2kgH73VKkIPVA.png)

Formally, the previous graphic would be:

![](https://miro.medium.com/max/60/1*9psJorlpFuAuj3lMKEQUFA.png?q=20)

![](https://miro.medium.com/max/700/1*9psJorlpFuAuj3lMKEQUFA.png)

# Supervised Naive Bayes Algorithm {#ca63}

The steps to perform in order to be able to use the Naive Bayes Algorithm to solve classification problems like the previous problem is:

1. Convert dataset into a frequency table
2. Creates a likelihood table by finding the probabilities of the events to occur.
3. The Naive Bayes equation is used to compute the posterior probability of each class.
4. The class with the higher posterior probability is the outcome of the prediction.

# Strengths and Weaknesses of Naive Bayes {#6b7b}

**The main strengths are:**

* Easy and quick way to predict classes, both in binary and multiclass classification problems.
* In the cases that the independence assumption fits, the algorithm performs better compared to other classification models, even with less training data.
* The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This helps with problems derived from the curse of dimensionality and improve the performance.+

Whereas **the main disadvantages** of using this method **are:**

* Although they are pretty good classifiers, naive bayes are know to be poor estimators. So the probability that outputs from it shouldn’t be taken very seriously.
* The naive assumption of independence is very unlikely to match real-world data.
* When the test data set has a feature that has not been observed in the training se, the model will assign a 0 probability to it and will be useless to make predictions. One of the main methods to avoid this, is the smoothing technique, being the Laplace estimation one of the most popular ones.



