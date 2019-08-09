\_A **decision tree** is drawn upside down with its root at the top. \_In the image on the left, the bold text in black represents a condition/**internal node**, based on which the tree splits into branches/**edges**. The end of the branch that doesn’t split anymore is the decision/**leaf. **

**Entropy:** This is a measure of uncertainty. Entropy is a measure of how messy the data is.  Hence High knowledge means low entropy and low knowledge means high entropy. Decision trees are here to tidy the dataset by looking at the values of the feature vector associated with each data point. Based on the values of each feature, decisions are made that eventually leads to a leaf and an answer. At each step, each branching, you want to decrease the entropy, so this quantity is computed before the cut and after the cut. If it decreases, the split is validated and we can proceed to the next step, otherwise, we must try to split with another feature or stop this branch.

Products of probabilities are confusing because products of large number of probabilities is very tiny and  a small change in one of the factors could drastically alter their product. Hence sums are better than products. To convert to sum we use **log.**

**log\(ab\) = log\(a\) + log\(b\) **Hence:** **Entropy = Average\(-log\(P\(winning\)\)\)

Hence entropy for a bucket with m red balls and n blue balls.

![](/assets/Screenshot 2019-08-09 at 12.41.37 PM.png)![](/assets/Screenshot 2019-08-09 at 12.45.42 PM.png)More the information gain the better is we can divide the data more cleanly

### **Random Forest**

For large datasets with multiple features if we build a single decision tree using all features it can result in overfitting as it will divide every data based on information gain. To avoid this we randomly select some features and build a decision tree. Then we again do it with some more random columns. Once this is done we use the results of all decision trees and pick the decision that appeared the most number of times.

