## Neural Network Building Blocks

## ![](/assets/Screenshot 2019-08-12 at 9.04.18 AM.png)![](/assets/Screenshot 2019-08-12 at 9.07.31 AM.png)![](/assets/Screenshot 2019-08-12 at 9.16.37 AM.png)![](/assets/Screenshot 2019-08-12 at 9.18.26 AM.png)Error Function

#### Gradient Descent

###### log-loss error function

#### The error function should be continuous and differentiable![](/assets/Screenshot 2019-08-12 at 9.31.19 AM.png)![](/assets/Screenshot 2019-08-12 at 9.36.24 AM.png)![](/assets/Screenshot 2019-08-12 at 9.38.03 AM.png)![](/assets/Screenshot 2019-08-12 at 9.38.38 AM.png)

#### Softmax function\(Used to classify when there is more that 2 choices\)

Softmax function, _activation function_ that turns numbers aka logits into probabilities that sum to one. Softmax function outputs a vector that represents the probability distributions of a list of potential outcomes

**exp function turns every number into a positive number. This is to take care of negative scores of linear functions**

#### ![](/assets/Screenshot 2019-08-12 at 9.48.09 AM.png)One hot encoding

#### ![](/assets/Screenshot 2019-08-12 at 9.49.54 AM.png)Maximum Likelihood \(Importance of probabilities\)

To maximize the probability in the left figure to meet the probability on the right so that it classifies properly is maximum likelihood. The model classifies most points correctly with P\(all\) indicating how accurate the model is. **Hence maximizing the probability is minimizing the error function**.![](/assets/Screenshot 2019-08-12 at 9.53.26 AM.png)**log function turns products into sum**

#### Cross Entropy\(Error function\)

Converting the above product of probabilities to sum using log. A good model with give a low cross entropy while a bad model will give a high entropy. A higher cross entropy implies a lower probability of the event.

##### ![](/assets/Screenshot 2019-08-12 at 10.02.55 AM.png)![](/assets/Screenshot 2019-08-12 at 10.03.56 AM.png)**Goal: Minimize the Cross Entropy**

![](/assets/Screenshot 2019-08-12 at 10.12.25 AM.png)![](/assets/Screenshot 2019-08-12 at 10.09.40 AM.png)**Multi class cross entropy**

#### ![](/assets/Screenshot 2019-08-12 at 10.16.26 AM.png)Logistic Regression

![](/assets/Screenshot 2019-08-12 at 10.31.17 AM.png)**Binary classification**![](/assets/Screenshot 2019-08-12 at 10.32.46 AM.png)**Multiclass error function**![](/assets/Screenshot 2019-08-12 at 10.33.43 AM.png)To minimize start with random weights and use gradient descent:

#### ![](/assets/Screenshot 2019-08-12 at 10.38.11 AM.png)Gradient Descent

![](/assets/Screenshot 2019-08-12 at 10.39.40 AM.png)![](/assets/Screenshot 2019-08-12 at 10.41.36 AM.png)![](/assets/Screenshot 2019-08-12 at 10.46.52 AM.png)![](/assets/Screenshot 2019-08-12 at 10.54.51 AM.png)

