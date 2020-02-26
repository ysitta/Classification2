# Classification 2 Quiz

This quiz is part of Algoritma Academy assessment process. Congratulations on completing the second Classification in Machine Learning course! We will conduct an assessment quiz to test practical classification model techniques you have learned on the course. The quiz is expected to be taken in the classroom, please contact our team of instructors if you missed the chance to take it in class.

In this quiz, you will use the loan dataset — the data stored as a csv format in this repository as a `loan.csv` file. To complete this assignment, you will need to build your classification model using naive bayes, decision tree, and random forest algorithms by following these steps:

# Data Exploration

Before we jump into modelling, we will start with data exploration first. Hence, load and investigate the data given `loan.csv` and assign it to the loan object, followed by investigating the data using `str()` or `glimpse()` function.

```
# your code here

```

Based on our investigation above, the loan data consists of 1000 observations and 17 variables, *which shows historical data of customers who are likely to default or not in a bank*. Meanwhile, the description of each feature explained below:

- `checking_balance and savings_balance`: Status of existing checking/savings account
- `credit_history`: Between critical, good, perfect, poor and very good
- `purpose`: Between business, car(new), car(used), education, furniture, and renovations
- `employment_duration`: Present employment since
- `percent_of_income`: Installment rate in percentage of disposable income
- `years_at_residence`: Present residence since
- `other_credit`: Other installment plans (bank/store)
- `housing`: Between rent, own, or for free
- `job`: Between management, skilled, unskilled and unemployed
- `dependents`: Number of people being liable to provide maintenance for
- `phone`: Between none and yes (registered under customer name)

As a data scientist, you will develop a model that aids management with their decision-making process. The first thing we need to know is what kind of business question we would like to solve. Loans are risky but at the same time it is also a product that generates profits for the institution through differential borrowing/ lending rates. So identifying risky customers is one way to minimize lender losses. From there, we will try to predict using the given set of predictors and how we model the default variable.

Before go through the modeling section, take your time to do the exploration step. Let's say we are instructed to investigate the historical number of default customers for each loan purpose. Please do some data aggregation to get the answer.

*Hint: Because we only focused at the debtor/borrower who defaulted, filter the historical data with the condition needed first (default == "yes")*

```
# your code here

```

___
1. Based on the exploration above, which purpose is most often to default?
  - [ ] furniture/appliances
  - [ ] car
  - [ ] business
  - [ ] education
___


# Cross-Validation

Before we build our model, we should split the dataset into training and test data. In this step, please split the data into 80% training and 20% test proportion using `sample()` function, `set.seed(100)`, and store it as `data_train` and `data_test`.

> Notes: Make sure you use RNGkind() before splitting

```
RNGkind(sample.kind = "Rounding")
set.seed(100)
# your code here

```

Let's take a look distribution of proportion in train and test data using `prop.table(table(object$target))` function to make sure in train and test data has balance or not distribution of each class target.

```
# your code here

```

Based on the proportion of the target variable above, we can conclude that our target variable can be considered to be imbalance; hence we will have to balance the same before using it for our models. One most important things to be kept in mind is that all sub-samping operations have to be applied to only training datasets. So please do it on `data_train` using the `downSample()` function from the caret package.

> Notes: set the argument `yname = "default"`

```
# your code here

```

# Decision Tree

After splitting our data into data_train and data_test, let us build our first model of a decision tree using `ctree()` function and store it under the `model_dt` object. To tune our model, let us set the `mincriterion = 0.10` argument.

```
# your code here

```
___
2. In our decision model, the goal of setting the `mincriterion = 0.10` is ...
  - [ ] To prune our model, we let the tree that has maximum p-value <= 0.10 to split the node
  - [ ] To prune our model, we let the tree that has maximum p-value <= 0.90 to split the node
  - [ ] To prune our model, we let the tree that has a maximum of 10% of the data in the terminal nodes
  - [ ] To prune our model, we let the tree that has a maximum of 10% of the data in the terminal nodes
___

To have a better grasp of our model, please try and plot the model using `type = "simple"` argument.

```
# your code here

```

# Predict on data train and test

The goal of a good machine learning model is to generalize well from the training data to any data from the problem domain. This allows us to make predictions in the future on data the model has never seen. There is a terminology used in machine learning when we talk about how well a machine learning model learns and generalizes to new data, namely overfitting and underfitting. So to validate whether our model is fit enough, please predict towards `data_train` and `data_test` based on `model_dt` using `predict()` function and choose the `type = "response"`.

```
# pred_train_dt <-
# pred_test_dt <-
```

# Model Evaluation

The last part of building the model would be the model evaluation. To check the performance, we can use the `confusionMatrix()` to get our model performance. Please create two confusion matrix using our in-sample data (`data_train`) and out-of-sample data (`data_test`) and compare both performances.
___
3. From the decision tree performance above, we can conclude that our decision tree model is ...
  - [ ] Overall balanced
  - [ ] Overfitting
  - [ ] Underfitting
___

# Random Forest

The second model that we want to build is Random Forest. Now, let us try to explore the random forest model we have prepared in `model_rf.RDS`. The `model_rf.RDS` is built with the following hyperparameter:

- `set.seed(100)` # the seed number
- `number = 5` # the number of k-fold cross-validation
- `repeats = 3` # the number of the iteration

In your environment, please load the random forest model (`model_rf.RDS`) and save it under the `model_rf` object using the `readRDS()` function.

```
# your code here

```

___
4. Which of the following are NOT among the advantages of the random forest model?
  - [ ] Automatic feature selection
  - [ ] It is a relatively fast model to compared to decision tree
  - [ ] Reduce bias in a model as it is the aggregate multiple decision trees
  - [ ] It generates an unbiased estimate of the out-of-box error
___

Now check the summary of the final model we built using `model_rf$finalModel`.

```
# your code here

```

In practice, the random forest already have out-of-bag estimates (OOB) that represent an unbiased estimate of its accuracy on unseen data.

___
5. Based on the `model_rf$finalModel` summary above, the out-of-bag error rate from our model is 33.61%, what does this mean?
  - [ ] We have error 33.61% of our unseen data
  - [ ] We have error 33.61% of our train data
  - [ ] We have error 33.61% of our loan data
  - [ ] We have error 33.61% of our in-sample data
___
  
# Predicting the test data
  
After building the model, we can now predict the data_train and data_test based on `model_rf`, from there, please choose the `type = "raw"`.

```
# your code here

```

# Model evaluation

Next, let us evaluate the random forest model we built with `confusionMatrix()` function and try to evaluate the performance of the random forest model. How should you evaluate the model performance?

```
# your code here

```

We could also use *Variable Importance*, to get a list of the most important variables used in our random forest. Many would argue that random forest, being a black box model, can offer no true information beyond its job in accuracy; actually paying special attention to attributes like variable importance for example often do help us gain valuable information about our data.

Please take your time try to check which variable has a high significance to the prediction. We can check it using `varImp()` function and pass into the `plot()` to get the visualization.

___
6. From the plot you have created, which is the most variable influence the default?
  - [ ] checking_balance
  - [ ] months_loan_duration
  - [ ] amount
  - [ ] purpose
___
  
# Naive Bayes

The last model we are trying to compare to is the Naive Bayes. There are several advantages in using this model, for example:

- The model is relatively fast to train
- It is estimating a probabilistic prediction
- It can handle irrelevant features

___
7. Below are the characteristics of Naive Bayes, **EXCEPT**
  - [ ] Assume that among the predictor variables are independent
  - [ ] Assume that between target and predictor variables are independent
  - [ ] Skewness due to data scarcity
___

# Naive Bayes model fitting

Now let us build a naive bayes model using `naiveBayes()` function from the `e1071` package, then set the laplace parameter as 1. Store the model under `model_naive` before moving on to the next section.

```
# your code here

```

# Predict the naive bayes model, using data train and data test

Using our test dataset we have created earlier, try to predict using `model_naive` and use `type = "raw"`. The prediction will then results in a probability of positive class happening for each test dataset. 

```
# your code here

```

Now, let's take a look at ROC and AUC performance, use the `prediction()` function from the `ROCR` package to compare the positive class in `pred_naive` (`pred_naive[,2]`) with the actual data (`data_test$default`) and store it as `pred_roc` object.

```
# your code here

```

Next, please use the performance function from the ROCR package to help us define the axes and assign it to a `perf` object. To use the performance function, please define the arguments as below:
  - `prediction.obj = pred_roc`
  - `measure = "tpr"`
  - `x.measure = "fpr"`

```
perf <-
```

After you created a `perf` object, plot the performance by passing it in `plot()` function.

```
# your code here
```

Try to evaluate the ROC Curve; see if there are any undesirable results from our model. Next, take a look at the AUC using `performance()` function, and set the arguments `prediction.obj = pred_roc`, and `measure = "auc"` then save it under auc object.

```
# your code here
```
___
8. From our naive bayes model above, how do you interpret the AUC value?
  - [ ] 78.44%, means the model performance is good because of the closer to 1 the better
  - [ ] 78.44%, means the model performance is weak because of the closer to 0 the better
  - [ ] 78.44%, the value of Area under ROC Curve did not give any information about model performance
___

# Model Evaluation Performance

Lastly, you can check the model performance for the naive bayes model using `confusionMatrix()` and compare it using the actual label.

```
# your code here
```

# Models Comparison

9. As a data scientist in a financial institution, we are required to generate a rule-based model that can be implemented to the existing system. What is the best model for us to pick?
  - [ ] Naive Bayes because all the conditional probabilities are well calculated
  - [ ] Decision Tree because a decision tree model is easily translatable to a set of rules
  - [ ] Random Forest because it is possible to traceback the rule using variable importance information
  
10. Between all of the models, which model has better performance in terms of identifying all high-risk customers?
  - [ ] Naive Bayes
  - [ ] Decision Tree
  - [ ] Random Forest
  