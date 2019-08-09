\_A **decision tree** is drawn upside down with its root at the top. \_In the image on the left, the bold text in black represents a condition/**internal node**, based on which the tree splits into branches/**edges**. The end of the branch that doesn’t split anymore is the decision/**leaf. **

**Entropy:** This is a measure of uncertainty. Entropy is a measure of how messy the data is.  Hence High knowledge means low entropy and low knowledge means high entropy. Decision trees are here to tidy the dataset by looking at the values of the feature vector associated with each data point. Based on the values of each feature, decisions are made that eventually leads to a leaf and an answer. At each step, each branching, you want to decrease the entropy, so this quantity is computed before the cut and after the cut. If it decreases, the split is validated and we can proceed to the next step, otherwise, we must try to split with another feature or stop this branch.

Products of probabilities are confusing because products of large number of probabilities is very tiny and  a small change in one of the factors could drastically alter their product. Hence sums are better than products. To convert to sum we use **log.**

**log\(ab\) = log\(a\) + log\(b\) **Hence:** **Entropy = Average\(-log\(P\(winning\)\)\)

Hence entropy for a bucket with m red balls and n blue balls.

![](/assets/Screenshot 2019-08-09 at 12.41.37 PM.png)![](/assets/Screenshot 2019-08-09 at 12.45.42 PM.png)More the information gain the better is we can divide the data more cleanly

### **Random Forest**

For large datasets with multiple features if we build a single decision tree using all features it can result in overfitting as it will divide every data based on information gain. To avoid this we randomly select some features and build a decision tree. Then we again do it with some more random columns. Once this is done we use the results of all decision trees and pick the decision that appeared the most number of times.

### **Hyperparameters**

##### Minimum number of samples to split

A node must have at least min\_samples_\__split samples in order to be large enough to split. If a node has fewer samples than min\_samples_\__split samples, it will not be split, and the splitting process stops.

##### Minimum number of samples per leaf

When splitting a node, one could run into the problem of having 99 samples in one of them, and 1 on the other. This will not take us too far in our process, and would be a waste of resources and time. If we want to avoid this, we can set a minimum for the number of samples we allow on each leaf.

If a threshold on a feature results in a leaf that has fewer samples than`min_samples_leaf`, the algorithm will not allow_that_split, but it may perform a split on the same feature at a_different threshold_, that_does_satisfy`min_samples_leaf`.

| small maximum depth | **Underfitting** |
| :--- | :--- |
| Large maximum depth | Overfitting |
| Small minimum samples per split | Overfitting |
| Large min samples per split | Underfitting |

[  
](https://classroom.udacity.com/nanodegrees/nd009t/parts/f873e6c9-5147-4e90-abc7-084afe9da5a1/modules/25f6d67e-fd75-4e4b-8fc1-20fc46674546/lessons/a31de5d2-25c3-4fdb-b4f4-5a15770ff888/concepts/a750d064-6240-47e7-87de-6e41dab807c5#)

