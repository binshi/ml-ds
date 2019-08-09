## Support Vector Machines

##### ![](/assets/Screenshot 2019-08-09 at 5.51.40 PM.png)![](/assets/Screenshot 2019-08-09 at 5.54.49 PM.png)![](/assets/Screenshot 2019-08-09 at 5.59.15 PM.png)Error = Classification Error + Margin Error              ==&gt;  Minimize using gradient descent

#### The C Paramter:

C is just a constant that attaches itself to the classification error.

##### Error = C\*Classification Error + Margin Error

![](/assets/Screenshot 2019-08-09 at 6.07.52 PM.png)

## Kernel Trick:

Linearly non-separable features often become linearly separable after they are mapped to a high dimensional feature space. Kernels are functions that help us map to higher dimensional space to classify the data. The degree of a polynomial is the hyperparameter we train to find the best possible model.

Think in Higher dimensions\(with multiple planes\) or use a circle.![](/assets/Screenshot 2019-08-09 at 6.10.42 PM.png)

Polynomial kernel![](/assets/Screenshot 2019-08-09 at 6.16.22 PM.png)![](/assets/Screenshot 2019-08-09 at 6.20.32 PM.png)

### RBF kernels\(Radial Basis Functions Kernel\)

### ![](/assets/Screenshot 2019-08-09 at 6.32.12 PM.png)![](/assets/Screenshot 2019-08-09 at 6.34.31 PM.png)

#### Usage

Linear kernels are good for text and they are needed for performance if you have a lot of data.

RBF kernels are general purpose usually the first thing to try if you are not processing text.

Other kernels may be good in very specific situations but are rare in practice. You should find quickly if a special kernel is good if you have very specific data like graphs or strings.

\[Picking a Kernel is \(mostly\) equivalent to picking a Regularizer \(or Regularization Operator\) and I will use them interchangeably\]

If you are classifying images, you can try a RBF Kernel--because the RBF Kernel selects solutions that are smooth \(this can be easily shown in frequency space.

If you think your solutions are naturally sparse, then pick an L1-regularizer.

If you only have a small set of labels but lots of unlabeled data, then you might try a Manifold Regularizer \(i.e. Transductive SVM\), with or without a non-linear Kernel

If you have text, then don't apply RBF. That makes no mathematical sense.

If you data lives on a graph, you might want a Diffusion Kernel.

if you can't find a Kernel or Regularizer that represents your problem, then build the features yourself, and run a L1-SVM.

[https://www.quora.com/Why-does-RBF-kernel-generally-outperforms-linear-or-polynomial-kernels/answer/Ferenc-Husz%C3%A1r?utm\_medium=organic&utm\_source=google\_rich\_qa&utm\_campaign=google\_rich\_qa](https://www.researchgate.net/deref/https%3A%2F%2Fwww.quora.com%2FWhy-does-RBF-kernel-generally-outperforms-linear-or-polynomial-kernels%2Fanswer%2FFerenc-Husz%25C3%25A1r%3Futm_medium%3Dorganic%26utm_source%3Dgoogle_rich_qa%26utm_campaign%3Dgoogle_rich_qa)

[https://www.quora.com/Why-does-RBF-kernel-generally-outperforms-linear-or-polynomial-kernels?utm\_medium=organic&utm\_source=google\_rich\_qa&utm\_campaign=google\_rich\_qa](https://www.researchgate.net/deref/https%3A%2F%2Fwww.quora.com%2FWhy-does-RBF-kernel-generally-outperforms-linear-or-polynomial-kernels%3Futm_medium%3Dorganic%26utm_source%3Dgoogle_rich_qa%26utm_campaign%3Dgoogle_rich_qa)

[https://stackoverflow.com/questions/27103456/linear-kernel-vs-rbf-kernel](https://www.researchgate.net/deref/https%3A%2F%2Fstackoverflow.com%2Fquestions%2F27103456%2Flinear-kernel-vs-rbf-kernel)

[https://www.ijraset.com/fileserve.php?FID=3040](https://www.researchgate.net/deref/https%3A%2F%2Fwww.ijraset.com%2Ffileserve.php%3FFID%3D3040)

[https://www.allanswered.com/post/1565/c-supervised-modeling-svm-linear-vs-rbf-kernels/](https://www.researchgate.net/deref/https%3A%2F%2Fwww.allanswered.com%2Fpost%2F1565%2Fc-supervised-modeling-svm-linear-vs-rbf-kernels%2F)

[https://www.csie.ntu.edu.tw/~cjlin/papers/kernel-check/kcheck.pdf](https://www.researchgate.net/deref/https%3A%2F%2Fwww.csie.ntu.edu.tw%2F~cjlin%2Fpapers%2Fkernel-check%2Fkcheck.pdf)

[https://data-flair.training/blogs/svm-kernel-functions/](https://www.researchgate.net/deref/https%3A%2F%2Fdata-flair.training%2Fblogs%2Fsvm-kernel-functions%2F)

[https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XL-2-W3/281/2014/isprsarchives-XL-2-W3-281-2014.pdf](https://www.researchgate.net/deref/https%3A%2F%2Fwww.int-arch-photogramm-remote-sens-spatial-inf-sci.net%2FXL-2-W3%2F281%2F2014%2Fisprsarchives-XL-2-W3-281-2014.pdf)

