---
id: 0
title: "Kaggle, WiDS Datathon: Extreme Weather Forcasting Contest"
subtitle: "Long range extreme weather forcasting for fighting climate change"
date: "2023.03.17"
tags: "Data Science, Weather Forcasting"
---

_**Background**_

The WiDS Datathon 2023 focuses on a prediction task involving forecasting sub-seasonal temperatures (temperatures over a two-week period, in our case) within the United States. They used a pre-prepared dataset consisting of weather and climate information for a number of US locations, for a number of start dates for the two-week observation, as well as the forecasted temperature and precipitation from a number of weather forecast models. Each row in the data corresponds to a single location and a single start date for the two-week period. The task is to predict the arithmetic mean of the maximum and minimum temperature, for each location and start date.

_**Solution**_

My solution first preprocess the data. It first solve location discrepencies by scaling the longitude and latitutde, then one-hot encoding categorical data, splitting date feature into 3 numerical variables, filling missing values, removing outliers and power transforming skewed data. Finally, I run Adverserial Validation to check if for each variable the training data distribution matches the test data distribution, then removed a couple features where there is a concept drift.

For training I used an algorithm for gradient boosting on decision trees: [Catboost](https://arxiv.org/abs/1706.09516), optimized by Bayesian Optimization, which in one sentence picks the next parameter value based on the performance of previous parameter value. In the end I also visualized feature importance to gain further insights into the trained model.

Finally, the training RMSE for our model is 0.40107, 0.39942 for the validation set, and 1.222 for the test set. It's on top 50% in the competition.

_**Challenges**_

My challenges in building this solution is in data preprocessing. Specifically, I was learning the correct way to:

- Tranform skewed data
- Remove outliers
- Check if distributions of training set are similar to test set

The first challenge to transform skewed data is to determine whether it's feasible in this contest. If the training data is transformed to be used for training, then the test set also needs to be transformed before making predictions. After verifying that this is indeed the case, I made sure that the test set is given to us to be used. Then, I found the variables that's skewed, and investigated different transform methods, including Min Max Scaler, log and power transform, and used KDE plot to verify the findings and results. Because some of the data is negative, log transform can't be used, and power transform looked better.

For removing outliers, I used Tukey's method. I compared different implementations, and picked one that looked the most clean and intuitive to me.

For checking if distributions of training set are similar to test set, I tried the algorithm suggested in this [medium article](https://medium.com/@praveenkotha/how-to-find-whether-train-data-and-test-data-comes-from-same-data-distribution-9259018343b), except its last(4th) step is incorrect: a classifier needs to be trained using cross validation on the entire dataset, instead of the training data, because the label of the training data will be homogeneously 1. Based on this reason, I adopted [this implementation](https://www.kaggle.com/code/kooaslansefat/wids-2023-woman-life-freedom) of the algorithm and it works perfectly.

_**Future Steps**_

The data is well-preprocessed, however, the model isn't fine-tuned. One future step could be to fine-tune the model and maybe try different optimizer like Adam.

Secondly, there are more models that could fit this dataset. I can choose some of those, compare the results and use the essemble method to improve RMSE.

_**Source Code**_

Since a lot happened in the source code, I'm not attaching any code snippet here. If you are interested in checking it out, [here](https://github.com/tianyimasf/kaggle/blob/main/wids-datathon-tianyi-yukyung-and-irsa.ipynb) is the full code for this project.

That's all! Hope you enjoyed, and feel free to comment below if you have any thoughts!
