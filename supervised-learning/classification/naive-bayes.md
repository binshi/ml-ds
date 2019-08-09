## Naive bayes

Source: [https://towardsdatascience.com/naive-bayes-intuition-and-implementation-ac328f9c9718](https://towardsdatascience.com/naive-bayes-intuition-and-implementation-ac328f9c9718)

In a broad sense, Naive Bayes models are a special kind of classification machine learning algorithms. They are based on a statistical classification technique called ‘Bayes Theorem’, a formula that calculates a probability by counting the frequency of values and combinations of values in the historical data.

Naive Bayes model are called ‘naive’ algorithms becaused they make an assumption that the predictor variables are independent from each other. In other words, that the presence of a certain feature ina dataset is completely unrelated to the presence of any other feature.

They provide aneasy way to build accurate models with very good performance given their simplicity.

They do this by providing a way to calculate the ‘posterior’ probability of a certain event _A_ to occur_,_ given some probabilities of ‘prior’ events. In layman's terms, the Bayes theorem calculates the probability of an event occurring, based on certain other probabilities that are related to the event in question. It is composed of a prior\(the probabilities that we are aware of or that is given to us\) and the posterior\(the probabilities we are looking to compute using the priors\).

![](https://miro.medium.com/max/60/1*Lt8E08oxEnnRegLbNBzNAg.png?q=20)

![](https://miro.medium.com/max/700/1*Lt8E08oxEnnRegLbNBzNAg.png)

We assume the following:

`P(D)`is the probability of a person having Diabetes. It's value is`0.01`or in other words, 1% of the general population has diabetes\(Disclaimer: these values are assumptions and are not reflective of any medical study\).

`P(Pos)`is the probability of getting a positive test result.

`P(Neg)`is the probability of getting a negative test result.

`P(Pos|D)`is the probability of getting a positive result on a test done for detecting diabetes, given that you have diabetes. This has a value`0.9`. In other words the test is correct 90% of the time. This is also called the Sensitivity or True Positive Rate.

`P(Neg|~D)`is the probability of getting a negative result on a test done for detecting diabetes, given that you do not have diabetes. This also has a value of`0.9`and is therefore correct, 90% of the time. This is also called the Specificity or True Negative Rate.

The Bayes formula is as follows:

![](/assets/NaiveBayes.png)

* `P(A)`is the prior probability of A occurring independently. In our example this is`P(D)`. This value is given to us.

* `P(B)`is the prior probability of B occurring independently. In our example this is`P(Pos)`.

* `P(A|B)`is the posterior probability that A occurs given B. In our example this is`P(D|Pos)`. That is,**the probability of an individual having diabetes, given that, that individual got a positive test result. This is the value that we are looking to calculate.**

* `P(B|A)`is the likelihood probability of B occurring, given A. In our example this is`P(Pos|D)`. This value is given to us.

Putting our values into the formula for Bayes theorem we get:

`P(D|Pos) = P(D) * P(Pos|D) / P(Pos)`

The probability of getting a positive test result`P(Pos)`can be calculated using the Sensitivity and Specificity as follows:

`P(Pos) = [P(D) * Sensitivity] + [P(~D) * (1-Specificity))]`

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

Naive Bayes classifier calculates the probabilities for every factor. Then it selects the outcome with highest probability. This classifier assumes the features \(in this case we had words as input\) are independent. Hence the word naive.

It's best to understand this theorem using an example. Let's say you are a member of the Secret Service and you have been deployed to protect the Democratic presidential nominee during one of his/her campaign speeches. Being a public event that is open to all, your job is not easy and you have to be on the constant lookout for threats. So one place to start is to put a certain threat-factor for each person. So based on the features of an individual, like the age, sex, and other smaller factors like whether the person is carrying a bag, looks nervous, etc., you can make a judgment call as to whether that person is a viable threat.

If an individual ticks all the boxes up to a level where it crosses a threshold of doubt in your mind, you can take action and remove that person from the vicinity. Bayes Theorem works in the same way, as we are computing the probability of an event \(a person being a threat\) based on the probabilities of certain related events \(age, sex, presence of bag or not, nervousness of the person, etc.\).

One thing to consider is the independence of these features amongst each other. For example if a child looks nervous at the event then the likelihood of that person being a threat is not as much as say if it was a grown man who was nervous. To break this down a bit further, here there are two features we are considering, age AND nervousness. Say we look at these features individually, we could design a model that flags ALL persons that are nervous as potential threats. However, it is likely that we will have a lot of false positives as there is a strong chance that minors present at the event will be nervous. Hence by considering the age of a person along with the 'nervousness' feature we would definitely get a more accurate result as to who are potential threats and who aren't.

This is the 'Naive' bit of the theorem where it considers each feature to be independent of each other which may not always be the case and hence that can affect the final judgement.

In short, Bayes Theorem calculates the probability of a certain event happening \(in our case, a message being spam\) based on the joint probabilistic distributions of certain other events \(in our case, the appearance of certain words in a message\).

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

## Applications of Naive Bayes Algorithms

* **Real time Prediction: **
  Naive Bayes is an eager learning classifier and it is sure fast. Thus, it could be used for making predictions in real time.
* **Multi class Prediction: **
  This algorithm is also well known for multi class prediction feature. Here we can predict the probability of multiple classes of target variable.
* **Text classification/ Spam Filtering/ Sentiment Analysis:**
   Naive Bayes classifiers mostly used in text classification \(due to better result in multi class problems and independence rule\) have higher success rate as compared to other algorithms. As a result, it is widely used in Spam filtering \(identify spam e-mail\) and Sentiment Analysis \(in social media analysis, to identify positive and negative customer sentiments\)
* **Recommendation System: **
  Naive Bayes Classifier and
  [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
  together builds a Recommendation System that uses machine learning and data mining techniques to filter unseen information and predict whether a user would like a given resource or not



