https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c

https://keras.io/losses/

https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0

https://blog.algorithmia.com/introduction-to-loss-functions/

https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

It describes how far off the result your network produced is from the expected result - it indicates the magnitude of error your model made on its prediciton. You can then take that error and 'backpropagate' it through your model, adjusting its weights and making it get closer to the truth the next time around.

The goal of machine learning and deep learning is to**reduce the difference between the predicted output and the actual output**. This is also called as a**Cost function\(C\) or Loss function**. Cost functions are convex functions.

**Loss functions fall under four major category**:

**Regressive loss functions: **They are used in case of regressive problems, that is when the target variable is continuous. Most widely used regressive loss function is Mean Square Error. Other loss functions are:  
1. Absolute error — measures the mean absolute value of the element-wise difference between input;  
2. Smooth Absolute Error — a smooth version of Abs Criterion.

**Classification loss functions: **The output variable in classification problem is usually a probability value f\(x\), called the score for the input x. Generally, the magnitude of the score represents the confidence of our prediction. The target variable y, is a binary variable, 1 for true and -1 for false. On an example \(x,y\), the margin is defined as yf\(x\). The margin is a measure of how correct we are. Most classification losses mainly aim to maximize the margin. Some classification algorithms are:

1. Binary Cross Entropy 

2. Negative Log Likelihood

3. Margin Classifier

4. Soft Margin Classifier

**Embedding loss functions: **It deals with problems where we have to measure whether two inputs are similar or dissimilar. Some examples are:

1. L1 Hinge Error- Calculates the L1 distance between two inputs.

2. Cosine Error- Cosine distance between two inputs.

