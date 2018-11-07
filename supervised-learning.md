In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

**Example 1:**

Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.

We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

**Example 2**:

\(a\) Regression - Given a picture of a person, we have to predict their age on the basis of the given picture

\(b\) Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.



# Difference Between Classification and Regression in Machine Learning

by [Jason Brownlee](https://machinelearningmastery.com/author/jasonb/) on December 11, 2017 [https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/)

There is an important difference between classification and regression problems.

Fundamentally, classification is about predicting a label and regression is about predicting a quantity.

I often see questions such as:

> How do I calculate accuracy for my regression problem?

Questions like this are a symptom of not truly understanding the difference between classification and regression and what accuracy is trying to measure.

In this tutorial, you will discover the differences between classification and regression.

After completing this tutorial, you will know:

* That predictive modeling is about the problem of learning a mapping function from inputs to outputs called function approximation.
* That classification is the problem of predicting a discrete class label output for an example.
* That regression is the problem of predicting a continuous quantity output for an example.

Tutorial Overview

This tutorial is divided into 5 parts; they are:

1. Function Approximation
2. Classification
3. Regression
4. Classification vs Regression
5. Converting Between Classification and Regression Problems

## Function Approximation

Predictive modeling is the problem of developing a model using historical data to make a prediction on new data where we do not have the answer.

For more on predictive modeling, see the post:

* [Gentle Introduction to Predictive Modeling](https://machinelearningmastery.com/gentle-introduction-to-predictive-modeling/)

Predictive modeling can be described as the mathematical problem of approximating a mapping function \(f\) from input variables \(X\) to output variables \(y\). This is called the problem of function approximation.

The job of the modeling algorithm is to find the best mapping function we can given the time and resources available.

For more on approximating functions in applied machine learning, see the post:

* [How Machine Learning Algorithms Work](https://machinelearningmastery.com/how-machine-learning-algorithms-work/)

Generally, we can divide all function approximation tasks into classification tasks and regression tasks.

## Classification Predictive Modeling

Classification predictive modeling is the task of approximating a mapping function \(f\) from input variables \(X\) to discrete output variables \(y\).

The output variables are often called labels or categories. The mapping function predicts the class or category for a given observation.

For example, an email of text can be classified as belonging to one of two classes: “spam_“\_and “\_not spam_“.

* A classification problem requires that examples be classified into one of two or more classes.
* A classification can have real-valued or discrete input variables.
* A problem with two classes is often called a two-class or binary classification problem.
* A problem with more than two classes is often called a multi-class classification problem.
* A problem where an example is assigned multiple classes is called a multi-label classification problem.

It is common for classification models to predict a continuous value as the probability of a given example belonging to each output class. The probabilities can be interpreted as the likelihood or confidence of a given example belonging to each class. A predicted probability can be converted into a class value by selecting the class label that has the highest probability.

For example, a specific email of text may be assigned the probabilities of 0.1 as being “spam” and 0.9 as being “not spam”. We can convert these probabilities to a class label by selecting the “not spam” label as it has the highest predicted likelihood.

There are many ways to estimate the skill of a classification predictive model, but perhaps the most common is to calculate the classification accuracy.

The classification accuracy is the percentage of correctly classified examples out of all predictions made.

For example, if a classification predictive model made 5 predictions and 3 of them were correct and 2 of them were incorrect, then the classification accuracy of the model based on just these predictions would be:

| accuracy = correct predictions / total predictions \* 100     accuracy = 3 / 5 \* 100     accuracy = 60% |
| :--- |


An algorithm that is capable of learning a classification predictive model is called a classification algorithm.

## Regression Predictive Modeling

Regression predictive modeling is the task of approximating a mapping function \(f\) from input variables \(X\) to a continuous output variable \(y\).

A continuous output variable is a real-value, such as an integer or floating point value. These are often quantities, such as amounts and sizes.

For example, a house may be predicted to sell for a specific dollar value, perhaps in the range of $100,000 to $200,000.

* A regression problem requires the prediction of a quantity.
* A regression can have real valued or discrete input variables.
* A problem with multiple input variables is often called a multivariate regression problem.
* A regression problem where input variables are ordered by time is called a time series forecasting problem.

Because a regression predictive model predicts a quantity, the skill of the model must be reported as an error in those predictions.

There are many ways to estimate the skill of a regression predictive model, but perhaps the most common is to calculate the root mean squared error, abbreviated by the acronym RMSE.

For example, if a regression predictive model made 2 predictions, one of 1.5 where the expected value is 1.0 and another of 3.3 and the expected value is 3.0, then the RMSE would be:

| RMSE = sqrt\(average\(error^2\)\)    RMSE = sqrt\(\(\(1.0 - 1.5\)^2 + \(3.0 - 3.3\)^2\) / 2\)   RMSE = sqrt\(\(0.25 + 0.09\) / 2\)  RMSE = sqrt\(0.17\)  RMSE = 0.412 |
| :--- |


A benefit of RMSE is that the units of the error score are in the same units as the predicted value.

An algorithm that is capable of learning a regression predictive model is called a regression algorithm.

Some algorithms have the word “regression” in their name, such as linear regression and logistic regression, which can make things confusing because linear regression is a regression algorithm whereas logistic regression is a classification algorithm.

## Classification vs Regression

Classification predictive modeling problems are different from regression predictive modeling problems.

* Classification is the task of predicting a discrete class label.
* Regression is the task of predicting a continuous quantity.

There is some overlap between the algorithms for classification and regression; for example:

* A classification algorithm may predict a continuous value, but the continuous value is in the form of a probability for a class label.
* A regression algorithm may predict a discrete value, but the discrete value in the form of an integer quantity.

Some algorithms can be used for both classification and regression with small modifications, such as decision trees and artificial neural networks. Some algorithms cannot, or cannot easily be used for both problem types, such as linear regression for regression predictive modeling and logistic regression for classification predictive modeling.

Importantly, the way that we evaluate classification and regression predictions varies and does not overlap, for example:

* Classification predictions can be evaluated using accuracy, whereas regression predictions cannot.
* Regression predictions can be evaluated using root mean squared error, whereas classification predictions cannot.

## Convert Between Classification and Regression Problems

In some cases, it is possible to convert a regression problem to a classification problem. For example, the quantity to be predicted could be converted into discrete buckets.

For example, amounts in a continuous range between $0 and $100 could be converted into 2 buckets:

* Class 0: $0 to $49
* Class 1: $50 to $100

This is often called discretization and the resulting output variable is a classification where the labels have an ordered relationship \(called ordinal\).

In some cases, a classification problem can be converted to a regression problem. For example, a label can be converted into a continuous range.

Some algorithms do this already by predicting a probability for each class that in turn could be scaled to a specific range:

| quantity = min + probability \* range |
| :--- |


Alternately, class values can be ordered and mapped to a continuous range:

* $0 to $49 for Class 1
* $50 to $100 for Class 2

If the class labels in the classification problem do not have a natural ordinal relationship, the conversion from classification to regression may result in surprising or poor performance as the model may learn a false or non-existent mapping from inputs to the continuous output range.

## Further Reading

This section provides more resources on the topic if you are looking to go deeper.

* [Gentle Introduction to Predictive Modeling](https://machinelearningmastery.com/gentle-introduction-to-predictive-modeling/)
* [How Machine Learning Algorithms Work](https://machinelearningmastery.com/how-machine-learning-algorithms-work/)

## Summary

In this tutorial, you discovered the difference between classification and regression problems.

Specifically, you learned:

* That predictive modeling is about the problem of learning a mapping function from inputs to outputs called function approximation.
* That classification is the problem of predicting a discrete class label output for an example.
* That regression is the problem of predicting a continuous quantity output for an example.



