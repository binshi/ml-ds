Andrew Ng: [https://www.coursera.org/learn/machine-learning/home/welcome](https://www.coursera.org/learn/machine-learning/home/welcome)

Machine learning was defined in 90’s by_**Arthur Samuel**\_described as the,”_**it is a field of study that gives the ability to the computer for self-learn without being explicitly programmed**\_”, that means imbuing knowledge to machines without hard-coding it.

> _**“A computer algorithm/program is said to learn from performance measure P and experience E with some class of tasks T if its performance at tasks in T, as measured by P, improves with experience E.”**_  
> -Tom M. Mitchell.

Machine learning is mainly focused on the development of computer programs which can teach themselves to grow and change when exposed to new data. Machine learning studies algorithms for self-learning to do stuff. It can process massive data faster with the learning algorithm. For instance, it will be interested in learning to complete a task, make accurate predictions, or behave intelligently.

### Why we need Machine Learning:- {#8d56}

Data is growing day by day, and it is impossible to understand all of the data with higher speed and higher accuracy. More than 80% of the data is unstructured that is audios, videos, photos, documents, graphs, etc. Finding patterns in data on planet earth is impossible for human brains. The data has been very massive, the time taken to compute would increase, and this is where Machine Learning comes into action, to help people with significant data in minimum time.

Machine Learning is a sub-field of AI. Applying AI, we wanted to build better and intelligent machines. It sounds similar to a new child learning from itself. So in the machine learning, a new capability for computers was developed. And now machine learning is present in so many segments of technology, that we don’t even realise it while using it.

### Types of Machine Learning:- {#ae10}

Machine Learning mainly divided into three categories, which are as follows-

![](https://cdn-images-1.medium.com/max/1600/1*8OSHpISmR1l79yX4I234wg.jpeg)

Types of Machine Learning

### 1.Supervised Learning:- {#1b67}

_**Supervised Learning**\_is the first type of machine learning, in which_**labelled**\_data used to train the algorithms. In supervised learning, algorithms are trained using marked data, where the input and the output are known. We input the data in the learning algorithm as a set of inputs, which is called as Features, denoted by X along with the corresponding outputs, which is indicated by Y, and the algorithm learns by comparing its actual production with correct outputs to find errors. It then modifies the model accordingly. The raw data divided into two parts. The first part is for training the algorithm, and the other region used for test the trained algorithm.

![](https://cdn-images-1.medium.com/max/1600/1*CfsLNK1yyBWXoiAU2qsiig.png)

Supervised Machine Learning

Supervised learning uses the data patterns to predict the values of additional data for the labels. This method will commonly use in applications where historical data predict likely upcoming events. Ex:- It can anticipate when transactions are likely to be fraudulent or which insurance customer is expected to file a claim.

### Types of Supervised Learning:- {#23c9}

The Supervised Learning mainly divided into two parts which are as follows-

![](https://cdn-images-1.medium.com/max/1600/1*2sYOHg35XnTmYQW3ri5oqw.png)

Types of Supervised Learning

### 1.1.Regression:- {#530e}

**Regression **is a **statistical **measurement used in finance, investing and other disciplines that attempts to determine the strength of the relationship between one dependent variable \(usually denoted by Y\) and a series of other changing variables \(known as independent variables\).

_**Regression**\_is the type of Supervised Learning in which labelled data used, and this data is used to make predictions in a continuous form. The output of the input is always ongoing, and the graph is linear. Regression is a form of predictive modelling technique which investigates the relationship between a dependent variable\[_**Outputs**_\] and independent variable\[_**Inputs**\_\]. This technique used for forecasting the weather, time series modelling, process optimisation. Ex:- One of the examples of the regression technique is House Price Prediction, where the price of the house will predict from the inputs such as No of rooms, Locality, Ease of transport, Age of house, Area of a home.

#### Types of Regression Algorithms:- {#ed8a}

There are many Regression algorithms are present in machine learning, which will use for different regression applications. Some of the main regression algorithms are as follows-

#### 1.1.1.Simple Linear Regression:- {#1f5d}

In simple linear regression, we predict scores on one variable from the ratings on a second variable. The variable we are forecasting is called the criterion variable and referred to as Y. The variable we are basing our predictions on is called the predictor variable and denoted to as X.

#### 1.1.2.Multiple Linear Regression:- {#7e83}

Multiple linear regression is one of the algorithms of regression technique, and it is the most common form of linear regression analysis. As a predictive analysis, the multiple linear regression is used to explain the relationship between one dependent variable with two or more than two independent variables. The independent variables can be continuous or categorical.

#### 1.1.3.Polynomial Regression:- {#502f}

Polynomial regression is another form of regression in which the maximum power of the independent variable is more than 1. In this regression technique, the best fit line is not a straight line instead it is in the form of a curve.

#### 1.1.4.Support Vector Regression:- {#df81}

Support Vector Regression can be applied not only to regression problems, but it also used in the case of classification. It contains all the features that characterise maximum margin algorithm. Linear learning machine mapping leans a non-linear function into high dimensional kernel-induced feature space. The system capacity was controlled by parameters that do not depend on the dimensionality of feature space.

#### 1.1.5.Ridge Regression:- {#853e}

Ridge Regression is one of the algorithms used in Regression technique. It is a technique for analysing multiple regression data that suffer from multicollinearity. By the addition of a degree of bias to the regression calculates, it reduces the standard errors. The net effect will be to give calculations that are more reliable.

#### 1.1.6.Lasso Regression:- {#bee4}

Lasso regression is a type of linear regression that uses shrinkage. Shrinkage is where data values shrunk towards a central point, like the mean. The lasso procedure encourages simple, sparse models \(i.e. models with fewer parameters\). This particular type of regression is well-suited for models showing high levels of multicollinearity or when you want to automate certain parts of model selection, like variable selection/parameter elimination.

#### 1.1.7.ElasticNet Regression:- {#57d9}

Elastic net regression combined L1 norms \(LASSO\) and L2 norms \(ridge regression\) into a penalised model for generalised linear regression, and it gives it sparsity \(L1\) and robustness \(L2\) properties.

#### 1.1.8.Bayesian Regression:- {#5a6b}

Bayesian regression allows a reasonably natural mechanism to survive insufficient data or poorly distributed data. It will enable you to put coefficients on the prior and the noise so that the priors can take over in the absence of data. More importantly, you can ask Bayesian regression which parts \(if any\) of its fit to the data are it confident about, and which parts are very uncertain.

#### 1.1.9.Decision Tree Regression:- {#cef6}

Decision tree builds a form like a tree structure from regression models. It breaks down the data into smaller subsets and while an associated decision tree developed incrementally at the same time. The result is a tree with decision nodes and leaf nodes.

#### 1.1.10.Random Forest Regression:- {#8f5f}

Random Forest is also one of the algorithms used in regression technique, and it is very flexible, easy to use machine learning algorithm that produces, even without hyper-parameter tuning. Also, this algorithm widely used because of its simplicity and the fact that it can use for both regression and classification tasks. The forest it builds, is an ensemble of Decision Trees, most of the time trained with the “bagging” method.

### 1.2.Classification:- {#18ef}

Classification is the type of Supervised Learning in which labelled data can use, and this data is used to make predictions in a non-continuous form. The output of the information is not always continuous, and the graph is non-linear. In the classification technique, the algorithm learns from the data input given to it and then uses this learning to classify new observation. This data set may merely be bi-class, or it may be multi-class too. Ex:- One of the examples of classification problems is to check whether the email is spam or not spam by train the algorithm for different spam words or emails.

#### Types of Classification Algorithms:- {#b1e6}

There are many Classification algorithms are present in machine learning, which used for different classification applications. Some of the main classification algorithms are as follows-

#### 1.2.1.Logistic Regression/Classification:- {#191b}

Logistic regression falls under the category of supervised learning; it measures the relationship between the dependent variable which is categorical with one or more than one independent variables by estimating probabilities using a logistic/sigmoid function. Logistic regression can generally use where the dependent variable is Binary or Dichotomous. It means that the dependent variable can take only two possible values like “Yes or No”, “Living or Dead”.

#### 1.2.2.K-Nearest Neighbours:- {#2766}

KNN algorithm is one of the most straightforward algorithms in classification, and it is one of the most used learning algorithms. A majority vote of an object is classified by its neighbours, with the purpose being assigned to the class most common among its k nearest neighbours. It can also use for regression — output is the value of the object \(predicts continuous values\). This value is the average \(or median\) of the benefits of its k nearest neighbours.

#### 1.2.3.Support Vector Machines:- {#b7ba}

A Support Vector Machine is a type of Classifier, in which a discriminative classifier formally defined by a separating hyperplane. The algorithm outputs an optimal hyperplane which categorises new examples. In two dimensional space, this hyperplane is a line dividing a plane into two parts wherein each class lay on either side.

#### 1.2.4.Kernel Support Vector Machines:- {#f5bd}

Kernel-SVM algorithm is one the algorithms used in classification technique, and it is mathematical functions set that defined as the kernel. The purpose of the core is to take data as input and transform it into the required form. Different SVM algorithms use different types of kernel functions. These functions can be different types. For example linear and nonlinear functions, polynomial functions, radial basis function, and sigmoid functions.

#### 1.2.5.Naive Bayes:- {#d36a}

Naive Bayes is a type of Classification technique, which based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other function. Naive Bayes model is accessible to build and particularly useful for extensive datasets.

#### 1.2.6.Decision Tree Classification:- {#d6d1}

Decision tree makes classification models in the form of a tree structure. An associated decision tree incrementally developed and at the same time It breaks down a large data-set into smaller subsets. The final result is a tree with decision nodes and leaf nodes. A decision node \(e.g., Root\) has two or more branches. Leaf node represents a classification or decision. The first decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data.

#### 1.2.7.Random Forest Classification:- {#ea00}

Random Forest is a supervised learning algorithm. It creates a forest and makes it somehow casual. The wood it builds is an ensemble of Decision Trees, it most of the time the decision tree algorithm trained with the “bagging” method, which is a combination of learning models increases the overall result.

### 2.Unsupervised Learning:- {#12e6}

Unsupervised Learning is the second type of machine learning, in which unlabeled data are used to train the algorithm, which means it used against data that has no historical labels. What is being showing must figure out by the algorithm. The purpose is to explore the data and find some structure within. In unsupervised learning the data is unlabeled, and the input of raw information directly to the algorithm without pre-processing of the data and without knowing the output of the data and the data can not divide into a train or test data. The algorithm figures out the data and according to the data segments, it makes clusters of data with new labels.

![](https://cdn-images-1.medium.com/max/1600/1*d0uYbtEC-ykWTwPD5WcrqQ.png)

Unsupervised Machine Learning

This learning technique works well on transactional data. For example, it can identify segments of customers with similar attributes who can then be treated similarly in marketing campaigns. Or it can find the primary qualities that separate customer segments from each other. These algorithms are also used to segment text topics, recommend items and identify data outliers.

### Types of Unsupervised Learning:- {#0492}

The Unsupervised Learning mainly divided into two parts which are as follows-

### 2.1.Clustering:- {#157c}

Clustering is the type of Unsupervised Learning in which unlabeled data used, and it is the process of grouping similar entities together, and then the grouped data is used to make clusters. The goal of this unsupervised machine learning technique is to find similarities in the data point and group similar data points together and to figures out that new data should belong to which cluster.

#### Types of Clustering Algorithms:- {#e349}

There are many Clustering algorithms are present in machine learning, which is used for different clustering applications. Some of the main clustering algorithms are as follows-

#### 2.1.1.K-Means Clustering:- {#989a}

K-Means clustering is one of the algorithms of Clustering technique, in which similar data grouped in a cluster. K-means is an iterative clustering algorithm that aims to find local maxima in each iteration. It starts with K as the input which is how many groups you want to see. Input k centroids in random locations in your space. Now, with the use of the Euclidean distance method calculate the distance between data points and centroids, and assign data point to the cluster which is close to it. Recalculate the cluster centres as a mean of data points attached to it. Repeat until no further changes occur.

![](https://cdn-images-1.medium.com/max/1600/1*ZBkw7-KNoiXRXkTmDwyhZA.jpeg)

K-Means Clustering showing 3 clusters

#### 2.1.2.Hierarchical Clustering:- {#e92a}

Hierarchical clustering is one of the algorithms of Clustering technique, in which similar data grouped in a cluster. It is an algorithm that builds the hierarchy of clusters. This algorithm starts with all the data points assigned to a bunch of their own. Then two nearest groups are merged into the same cluster. In the end, this algorithm terminates when there is only a single cluster left. Start by assign each data point to its bunch. Now find the closest pair of the group using Euclidean distance and merge them into the single cluster. Then calculate the distance between two nearest clusters and combine until all items clustered into a single cluster.

![](https://cdn-images-1.medium.com/max/1600/1*1zZQhyLI-72u07_zCfR8Hg.png)

### 2.2.Dimensionality Reduction:- {#71e6}

Dimensionality Reduction is the type of Unsupervised Learning, in which the dimensions of the data is reduced to remove the unwanted data from the input. This technique is used to remove the undesirable features of the data. It relates to the process of converting a set of data having large dimensions into data with carries same data and small sizes. These techniques used while solving machine learning problems to obtain better features.

#### Types of Dimensionality Reduction Algorithms:- {#558e}

There are many Dimensionality reduction algorithms are present in machine learning, which applied for different dimensionality reduction applications. Some of the main dimensionality reduction algorithms are as follows-

#### 2.2.1.Principal Component Analysis:- {#dbd1}

Principal Component Analysis is one of the algorithms of Dimensionality Reduction, in this technique, it transformed into a new set of variables from old variables, which are the linear combination of real variables. Specific new set of variables are known as principal components. As a result of the transformation, the first primary component has the most significant possible variance, and each following element has the highest potential difference under the constraint that it is orthogonal to the above ingredients. Keeping only the first m &lt; n components reduces the data dimensionality while retaining most of the data information,

#### 2.2.2.Linear Discriminant Analysis:- {#29c1}

The linear discriminant analysis is one of the algorithms of Dimensionality Reduction in which it also creates linear combinations of your original features. However, unlike PCA, LDA doesn’t maximise explained variance. Instead, it optimises the separability between classes. LDA can improve the predictive performance of the extracted features. Furthermore, LDA offers variations to tackle specific roadblocks.

#### 2.2.3.Kernel Principal Component Analysis:- {#c7bf}

Kernel Principal Component Analysis is one of the algorithms of Dimensionality Reduction, and the variables which are transformed into variables of the new set, which are the non-linear combination of original variables means the nonlinear version of PCA, called as Kernel Principal Component Analysis \(KPCA\). It is capable of capturing part of the high order statistics, thus provides more information from the original dataset.

### 3.Reinforcement Learning:- {#2dcf}

Reinforcement Learning is the third type of machine learning in which no raw data is given as input instead reinforcement learning algorithm have to figures out the situation on their own. The reinforcement learning frequently used for robotics, gaming, and navigation. With reinforcement learning, the algorithm discovers through trial and error which actions yield the most significant rewards. This type of training has three main components which are the agent which can describe as the learner or decision maker, the environment which described as everything the agent interacts with and actions which represented as what the agent can do.

![](https://cdn-images-1.medium.com/max/1600/1*zStkxQGIbhIbz390BOE5Qw.png)

Reinforcement Learning

The objective is for the agent to take actions that maximise the expected reward over a given measure of time. The agent will reach the goal much quicker by following a good policy. So the purpose of reinforcement learning is to learn the best plan.

#### Types of Reinforcement Learning Algorithms:- {#7d7f}

There are many Reinforcement Learning algorithms are present in machine learning, which applied for different reinforcement learning applications. Some of the main algorithms are as follows-

#### 3.1.Q-Learning:- {#89db}

Q-learning is one of the algorithms of Reinforcement Learning, in which an agent attempts to learn the optimal strategy from its history of communication with the environment. A record of an agent is a sequence of state-action-rewards. Q-learning learns an optimal policy no matter which procedure the agent is following as long as there is no restriction on the plenty of times it tries an action in any state. Because it learns an optimal policy no matter which strategy it is carrying out, it is called an off-policy method.

#### 3.2.SARSA \[State Action Reward State Action\]:- {#b878}

SARSA is one of the algorithms of Reinforcement Learning, in which it determines it refreshed to the action values. It’s a minor difference between the SARSA and Q-learning implementations, but it causes a profound effect. The SARSA method takes another parameter, action2, which is the action that was made by the agent from the second state. It allows the agent to find the future reward value explicitly. Next, that followed, rather than assuming that the optimal action will use and that the most significant reward.

#### 3.3.Deep Q-Network:- {#5128}

Deep Q-Network is one of the algorithms of Reinforcement Learning, although Q-learning is a very robust algorithm, its main flaw is lack of generality. If you view Q-learning as renewing numbers in a two-dimensional array \(Action Space \* State Space\), it, in fact, follows the dynamic programming. It indicates that for states that the Q-learning agent has not seen before, it has no clue which action to take. In other words, a Q-learning agent cannot estimate value for unseen states. To deal with this problem, DQN gets rid of the two-dimensional array by introducing Neural Network.

#### 3.4.Markov Decision Processes:- {#8531}

Markov Decision Process is one of the algorithms of Reinforcement Learning, in which it contains \*A set of possible world states S. \*A set of Models. \*A set of possible actions A. \*A real-valued reward function R\(s, a\). \*A policy the solution of Markov Decision Process. To achieve a goal, the Markov Decision Process is used it is a straightforward framing of the problem of learning from interaction. The agent was selecting actions and the environment responding to these actions, and the agent and the environment interact continually and presenting new situations to the agent.

#### 3.5.DDPG\[Deep Deterministic Policy Gradient\]:- {#f2c7}

Deep Deterministic Policy Gradient is one of the algorithms of Reinforcement Learning, in which it relies on the actor-critic design with two eponymous components, actor, and critic. An actor is utilised to tune the parameter 𝜽 for the policy function, i.e. decide the best action for a specific state. The ideas of separate target network and experience replay are also borrowed from DQN. The seldom performs exploration for operations is another issue for DDPG. A solution for this is adding noise to the parameter space or the action space.

### 4.Semi-Supervised Learning:- {#2259}

Semi-Supervised Learning is the fourth type of Machine Learning, in which both types of raw data used. Semi-supervised learning is a hybrid of supervised and unsupervised machine learning. The Semi-supervised learning used for the same purposes as supervised learning, where it employs both labelled and unlabeled data for training typically a small amount of labelled data with a significant amount of unlabeled data. This type of learning can use with methods such as classification, regression, and prediction.

![](https://cdn-images-1.medium.com/max/1600/1*dILZZn_m_Tn3eRoNv_VBgQ.png)

Semi-supervised machine learning

This technique is useful for a few reasons. First, the process of labelling massive amounts of data for supervised learning is often prohibitively time-consuming and expensive. What’s more, too much labelling can impose human biases on the model. That means including lots of unlabeled data during the training process tends to improve the accuracy of the final model while reducing the time and cost spent building it.

### Applications of Machine Learning:- {#eef7}

There are many uses of Machine Learning in various fields, some of the areas are Medical, Defence, Technology, Finance, Security, etc. These fields areas different applications of Supervised, Unsupervised and Reinforcement learning. Some of the areas where these ML algorithms used are as follows-

![](https://cdn-images-1.medium.com/max/1600/1*OtfqDBQjP669bKFcAieCDA.png)

> source: [https://towardsdatascience.com/machine-learning-for-beginners-d247a9420dab](https://towardsdatascience.com/machine-learning-for-beginners-d247a9420dab)

## ![](/assets/Process-Overview.png)

##### Source: [https://machinelearningmastery.com/4-steps-to-get-started-in-machine-learning/](https://machinelearningmastery.com/4-steps-to-get-started-in-machine-learning/)

## What is Machine Learning?

Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications:

Supervised learning and Unsupervised learning.

# ![](/assets/MachineLearningAlgorithms.png)



