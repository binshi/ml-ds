# Appendix:

**Regression**: a measure of the relation between the mean value of one variable \(e.g. output\) and corresponding values of other variables \(e.g. time and cost\).

**probability**: A measure of uncertainty which lies between 0 and 1, where 0 means impossible and 1 means certain. Probabilities are often expressed as a percentages \(such as 0%, 50% and 100%\).

**random variable**: A variable \(a named quantity\) whose value is uncertain.

**normalization constraint**:The constraint that the[probabilities](http://mbmlbook.com/MurderMystery.html#probability)given by a[probability distribution](http://mbmlbook.com/MurderMystery.html#probability_distribution)must add up to 1 over all possible values of the[random variable](http://mbmlbook.com/MurderMystery.html#random_variable). For example, for aBernoulli\(p\)Bernoulli\(p\)distribution the[probability](http://mbmlbook.com/MurderMystery.html#probability)oftrueis ppand so the[probability](http://mbmlbook.com/MurderMystery.html#probability)of the only other statefalsemust be1−p1−p.

**probability distribution**: A function which gives the[probability](http://mbmlbook.com/MurderMystery.html#probability)for every possible value of a[random variable](http://mbmlbook.com/MurderMystery.html#random_variable). Written asP\(A\)P\(A\)for a[random variable](http://mbmlbook.com/MurderMystery.html#random_variable)A.

**sampling**: Randomly choosing a value such that the[probability](http://mbmlbook.com/MurderMystery.html#probability)of picking any particular value is given by a[probability distribution](http://mbmlbook.com/MurderMystery.html#probability_distribution). This is known as sampling from the distribution. For example, here are 10 samples from aBernoulli\(0.7\) distribution:false,true,false,false,true,true,true,false,trueandtrue. If we took a very large number of samples from aBernoulli\(0.7\) distribution then the percentage of the samples equal totruewould be very close to 70%.

**Bernoulli distribution**: A[probability distribution](http://mbmlbook.com/MurderMystery.html#probability_distribution)over a two-valued \(binary\)[random variable](http://mbmlbook.com/MurderMystery.html#random_variable). The Bernoulli distribution has one parameterppwhich is the[probability](http://mbmlbook.com/MurderMystery.html#probability)of the valuetrueand is written asBernoulli\(p\)\(p\). As an example,Bernoulli\(0.5\)\(0.5\)represents the uncertainty in the outcome of a fair coin toss.

**uniform distribution**: A[probability distribution](http://mbmlbook.com/MurderMystery.html#probability_distribution)where every possible value is equally probable. For example,Bernoulli\(0.5\)\(0.5\)is a uniform distribution sincetrueandfalseboth have the same[probability](http://mbmlbook.com/MurderMystery.html#probability)\(of 0.5\) and these are the only possible values.

**point mass**: A distribution which gives[probability](http://mbmlbook.com/MurderMystery.html#probability)1 to one value and[probability](http://mbmlbook.com/MurderMystery.html#probability)0 to all other values, which means that the[random variable](http://mbmlbook.com/MurderMystery.html#random_variable)is certain to have the specified value. For example,Bernoulli\(1\)Bernoulli\(1\)is a point mass indicating that the variable is certain to betrue.



  


