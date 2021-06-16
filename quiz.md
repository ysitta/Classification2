# Classification 2 Quiz

This quiz is part of Algoritma Academy assessment process. Congratulations on completing the second Classification in Machine Learning course! We will conduct an assessment quiz to test the practical classification model techniques that you have learned on the course. The quiz is expected to be taken in the classroom, please contact our team of instructors if you missed the chance to take it in class.

In this quiz, you will analyze the loan dataset which shows *historical data of customers who are likely to default or not in a bank*. The data stored in this repository as `loan.csv`. To complete this assignment, you will need to build classification models using Naive Bayes, Decision Tree, and Random Forest algorithms by following these steps:

# Data Exploration

Before we jump into modeling, we will try to explore the data. Load the data given (`loan.csv`) and assign it to an object named `loan`, followed by investigating the data using `str()` or `glimpse()` function.

```
# your code here

```

Based on our investigation above, the loan data consists of 1000 observations and 17 variables. The description of each feature explained below:

- `checking_balance` and `savings_balance`: Status of existing checking/savings account
- `months_loan_duration`: Duration of the loan period in months
- `credit_history`: Between critical, good, perfect, poor, and very good
- `purpose`: Between business, car(new), car(used), education, furniture, and renovations
- `amount`: Loan amount in DM (Deutsche Mark)
- `employment_duration`: Length of time at current job
- `percent_of_income`: Installment rate in percentage of disposable income
- `years_at_residence`: Number of years at current residence
- `age`: Customer's age
- `other_credit`: Other installment plans (bank/store)
- `housing`: Between rent, own, or for free
- `existing_loans_count`: Number of ongoing loans
- `job`: Between management, skilled, unskilled and unemployed
- `dependents`: Number of people being liable to provide maintenance for
- `phone`: Either no or yes (registered under customer name)
- `default`: Either no or yes. A loan's default is considered as yes when it is defaulted, charged off, or past due date

You should also make sure that each column store the right data types. You can do data wrangling below if you need to.

*Tips: You can also use parameter `stringsAsFactors = TRUE` from `read.csv()` so that all character column will automatically stored as factors.*

```
# your code

```

As a data scientist, you will develop a model that aids management with their decision-making process. The first thing we need to know is what kind of business question we would like to solve. Loans are risky but at the same time it is also a product that generates profits for the institution through differential borrowing/lending rates. So **identifying risky customers** is one way to minimize lender losses. From there, we will try to predict using the given set of predictors and how we model the `default` variable.

Before we go through the modeling section, take your time to do the exploration step. Try to investigate the historical number of defaulted customers for each loan purpose. Please do some data aggregation to get the answer.

*Hint: Because we only focused of the customers who defaulted, filter the data based on the condition needed (default == "yes")*

```
# your code here

```

___
1. Based on the exploration above, which purpose is most often to default?
  - [ ] Furniture/appliances
  - [ ] Car
  - [ ] Business
  - [ ] Education
___

# Cross-Validation

Before we build our model, we should split the dataset into training and test data. Please split the data into 80% training and 20% test using `sample()` function, `set.seed(100)`, and store it as `data_train` and `data_test`.

> Notes: Make sure you use `RNGkind()` and `set.seed()` before splitting and run them together with your `sample()` code

```
RNGkind(sample.kind = "Rounding")
set.seed(100)
# your code here

```

Let's look at the proportion of our target classes in train data using `prop.table(table(object$target))` to make sure we have a balanced proportion in train data.

```
# your code here

```

Based on the proportion above, we can conclude that our target variable can be considered imbalanced; hence we will have to balance the data before using it for our models. One important thing to be kept in mind is that all sub-sampling operations have to be applied only to training dataset. So please do it on `data_train` using the `downSample()` function from the caret package, and then store the downsampled data in `data_train_down` object. You also need to make sure that the target variable already stored in factor data type.

> Notes: set the argument `yname = "default"`

```
library(caret)
set.seed(100)
# your code here
data_train_down <- 


```

> In the following step, please use `data_train_down` to build Naive Bayes, Decision Tree, and Random Forest model below.

# Naive Bayes

After splitting our data into train and test set and downsample our train data, let us build our first model of Naive Bayes. There are several advantages in using this model, for example:

- The model is relatively fast to train
- It is estimating a probabilistic prediction
- It can handle irrelevant features

___
2. Below are the characteristics of Naive Bayes, **EXCEPT**
  - [ ] Assume that among the predictor variables are independent
  - [ ] Assume that between target and predictor variables are independent
  - [ ] Skewness due to data scarcity
___

Build a Naive Bayes model using `naiveBayes()` function from the `e1071` package, then set the laplace parameter as 1. Store the model under `model_naive` before moving on to the next section.

```
library(e1071)
# your code here
model_naive <- 
```

# Naive Bayes Model Prediction

Try to predict our test data using `model_naive` and use `type = "class"` to obtain class prediction. Store the prediction under `pred_naive` object.

```
# your code here
pred_naive <- 
```

# Naive Bayes Model Evaluation

The last part of model building would be the model evaluation. You can check the model performance for the Naive Bayes model using `confusionMatrix()` and compare the predicted class (`pred_naive`) with the actual label in `data_test`. Make sure that you're using defaulted customer as the positive class (`positive = "yes"`).

```
# your code here
```

# Decision Tree

The next model we're trying to build is Decision Tree. Use `ctree()` function to build the model and store it under the `model_dt` object. To tune our model, let's set the parameter `mincriterion = 0.90`.

```
library(partykit)
set.seed(100)
# your code here
model_dt <-
```
___
3. In our decision tree model, the goal of setting the `mincriterion = 0.90` is ...
  - [ ] To prune our model, we let the tree that has p-value <= 0.90 to split the node
  - [ ] To prune our model, we let the tree that has p-value <= 0.10 to split the node
  - [ ] To prune our model, we let the tree that has a maximum of 10% of the data in the terminal nodes

___

To have a better grasp of our model, please try to plot the model and set `type = "simple"`.

```
# your code here

```
___

4. Based on the plot, which of the following interpretation is **TRUE**?
  - [ ] a customer who has `checking_balance` > 200 DM, with `credit_history` labelled "perfect", and `saving_balance` that is "unknown" is expected to default
  - [ ] a customer who has `checking_balance` 1-200 DM, with `months_loan_duration` < 21 is expected to default
  - [ ] a customer who has `checking_balance` that is "unknown", with `other_credit` consist of "store" is expected to default
___

# Decision Tree Model Prediction

Now that we have the model, please predict towards the test data based on `model_dt` using `predict()` function and set the parameter `type = "response"` to obtain class prediction.

```
# your code here
pred_dt <- 
```

# Decision Tree Model Evaluation

We can use the `confusionMatrix()` to get our model performance. Make sure that you're using defaulted customer as the positive class (`positive = "yes"`). 

```
# your code here

```

# Random Forest

The last model that we want to build is Random Forest. he following are among the advantages of the random forest model:

- Reduce bias in a model as it aggregates multiple decision trees
- Automatic feature selection
- It generates an unbiased estimate of the out-of-box error
  
Now, let's explore the random forest model we have prepared in `model_rf.RDS`. The `model_rf.RDS` was built with the following hyperparameter:

- `set.seed(100)` # the seed number
- `number = 5` # the number of k-fold cross-validation
- `repeats = 3` # the number of the iteration

In your environment, please load the random forest model (`model_rf.RDS`) and save it under the `model_rf` object using the `readRDS()` function.

```
# your code here
model_rf <-
```

Now check the summary of the final model we built using `model_rf$finalModel`.

```
library(randomForest)
# your code here

```

In practice, the random forest already have an out-of-bag estimates (OOB) that represent its accuracy on out-of-bag data (the data that is not sampled/used for building random forest).

___
5. Based on the `model_rf$finalModel` summary above, how can we interpret the out-of-bag error rate from our model?
  - [ ] We have 33.61% error of our unseen data
  - [ ] We have 33.61% error of our train data
  - [ ] We have 33.61% error of our loan data
  - [ ] We have 33.61% error of our in-sample data
___

We could also use *Variable Importance*, to get a list of the most important variables used in our random forest. Many would argue that random forest, being a black box model, can offer no true information beyond its job in accuracy; actually paying special attention to attributes like variable importance for example often do help us gain valuable information about our data.

Please take your time to check which variable has a high influence to the prediction. You can use `varImp()` function and pass it to the `plot()` function to get the visualization.

```
# your code here

```

___
6. From the plot you have created, which variable has the most influence to the prediction?
  - [ ] checking_balance
  - [ ] months_loan_duration
  - [ ] amount
  - [ ] purpose
___ 
  
# Random Forest Model Prediction
  
After building the model, we can now predict the test data based on `model_rf`. You can use `predict()` function and set the parameter `type = "raw"` to obtain class prediction. 

```
# your code here
pred_rf <- 
```

# Random Forest Model Evaluation

Next, let us evaluate the random forest model we built using `confusionMatrix()`. How should you evaluate the model performance?

```
# your code here

```

Another way of evaluating model performance is through its ROC and AUC value. To calculate it, we need *the probability of a positive class for each observation*. Let's try focusing on the ROC and AUC value from our random forest prediction. First, predict the test data using `model_rf` but now using the parameter `type = "prob"`. The prediction will results in the probability values for each class. You can store the prediction in `prob_test` object.

```
# your code here
prob_test <- 
```

Now, use the `prediction()` function from the `ROCR` package to compare the *probability of positive class* in `prob_test[,"yes"]` with the actual data `data_test$default` and store it as `pred_roc` object.

```
# your code here
library(ROCR)
pred_roc <- 
```

Next, please use the `performance()` function from the ROCR package, define the axes, and assign it to a `perf` object. To use the `performance()` function, please define the arguments as below:
  - `prediction.obj = pred_roc`
  - `measure = "tpr"`
  - `x.measure = "fpr"`

```
# your code here
perf <- 
```

After you created a `perf` object, plot the performance by passing it in the `plot()` function.

```
# your code here

```

Try to evaluate the ROC Curve; see if there is any undesirable results from our model. Next, take a look at the AUC value using `performance()` function by setting the arguments `prediction.obj = pred_roc` and `measure = "auc"` then save it under `auc` object.

```
# your code here
auc <-
```

___
7. From the result above, how do you interpret the AUC value?
  - [ ] 90.51% means that the model performance is good because the closer to 1 the better
  - [ ] 90.51% means that the model performance is good in classifying positive classes
  - [ ] 95.11% means that the model performance is good in classifying both positive and negative class
  - [ ] 95.11% as Area under ROC Curve represent the accuracy of the model
___

# Models Comparison

___
8. As a data scientist in a financial institution, we are required to generate a rule-based model that can be easily implemented in the existing system. What is the best model for us to pick?
  - [ ] Naive Bayes because all the conditional probabilities are well calculated
  - [ ] Decision Tree because the model can be easily translated into a set of rules
  - [ ] Random Forest because it is possible to traceback the rule using variable importance information
  
9. Between all the models we have made, which model has better performance in terms of identifying all high-risk customers?
  - [ ] Naive Bayes
  - [ ] Decision Tree
  - [ ] Random Forest
___

Last but not least, The goal of a good machine learning model is to generalize well from the training data to any data from the problem domain. This allows us to make predictions in the future on the data the model has never seen. There is a terminology used in machine learning when we talk about how well a machine learning model learns and generalizes to new data, namely *overfitting* and *underfitting*. 

To validate whether our model is fit enough, we can predict the train and test data and then evaluate model performance in both data. You can check whether the performance is well balanced based on the threshold you have set.

___
10. Based on your knowledge about the characteristic of a machine learning model, which statement below is **FALSE**? 
  - [ ] Overfitting is a condition where a model performs well on the training data but performs very poorly in test data.
  - [ ] Underfitting is a condition where a model performs poor in the training data but performs well on the test data.
  - [ ] Machine Learning model that fit just right may have a slightly lower performance in its test data than in its training data.
___
