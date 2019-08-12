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



