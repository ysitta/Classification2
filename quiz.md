Congratulation! This is the end of Classification in Machine Learning 2. The last part of this course is closed by quiz assessment. In this quiz, you will use the loan dataset. The data is stored as csv file in this repository as loan.csv file. The dataset is downloaded from [UCI Machine Learning](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)). 

To complete this assignment, you will need to build your classification model using naive bayes, decision tree and random forest algorithms by following these steps:

# 1 Data Exploration

Please load and investigate the data given `loan.csv` and assign it to `loan` object, then investigate the data using `str()` or `glimpse()` function.

The loan data consists of 1000 observations and  17 variables. Loan dataset is a bank data that shows historical data of customer who are likely to default or not. The description of each features is explained below:

- `checking_balance and savings_balance`: Status of existing checking / savings account
- `credit_history`: Between critical, good, perfect, poor and very good
- `purpose`: Between business, car(new), car(used), education, furniture and renovations
- `employment_duration`: Present employment since
- `percent_of_income`: Installment rate in percentage of disposable income
- `years_at_residence`: Present residence since
- `other_credit`: Other installment plans (bank / store)
- `housing`: Between rent, own, or for free
- `job`: Between management, skilled, unskilled and unemployed
- `dependents`: Number of people being liable to provide maintenance for
- `phone`: Between none and yes (registered under customer name)

Let's say for example we are instructed to investigate the number of default customers for each loan purpose. Let us do some data aggregation to get the answer.

1. Based on the exploration above, what is the highest number of purpose to not default?
  - [ ] furniture/applicances
  - [ ] car
  - [ ] business
  - [ ] education

## 1.1 Know your business question

As a data scientist you will develop a model that aid management in making decision. First thing we need to know is what kind of business question that we want to solve. Based on the data our data, let us identify a high risk customers. We will try to predict if given a set of predictors, how does we model `default` variable.

# 2 Cross Validation

Before we build our model, we should split the dataset into train and test data. In this step, please split the data into 80% train and 20% test proportion use `sample()`function, `set.seed(100)`, and store it as `data_train` and `data_test`.

## 2.1 Check the proportion of our target variable

To make sure we have splitted the class target properly, let's check the proportion of `data_train$default` using `table()` and `prop.table()` function.

Based on the proportion of the target variable above, we can conclude that our target variable can be considered to be imblance, hence we should try to down sample it first. Please balance the proportion of the target variable on `data_train` using `downSample()` function from `caret` package.

# 3 Decision Tree

## 3.1 Decision Tree Model Fitting

After we have splitted our data into `data_train` and `data_test`, let's build our first model of decision tree using `ctree()` function and store it under `model_dt` object. To tune our model, let us set the `mincriterion = 0.10` parameter. 

## 3.2 Decision Tree Quiz

2. In our decision model, the goals of setting the mincriterion = 0.10 is to ...
  - [ ] To prune our model, we let the tree that has maximum p-value <= 0.10 to split the node
  - [ ] To prune our model, we let the tree that has maximum p-value <= 0.90 to split the node
  - [ ] To prune our model, we let the tree that has maximum 10% of the data in the terminal nodes
  - [ ] To prune our model, we let the tree that has maximum 10% of the data in the terminal nodes

To get a picture of our model, try to plot the model, and use the `type = "simple"` parameter.

## 3.3 Predict on data train and test

To get our prediction, please predict the model towards `data_train` and `data_test` based on `model_dt` using `predict()` function and choose the `type = "response"`

```
# pred_train_dt <-
# pred_test_dt <-
```

# 3.4 Model Evaluation

The last part of build the model will be evaluating our model performance. To do this, we can use the `confusionMatrix()` in order to get your model performance. Create two confusion matrix using our in-sample data (`data_train`) and out-of-sample data (`data_test`) you have prepared earlier, and compared both performance.

3. From the decision tree performance above, we can conclude that our decision tree model is ...
  - [ ] Overall balanced
  - [ ] Overfitting
  - [ ] Underfitting


# 4 Random Forest
## 4.1 Random forest model fitting

Now let's try to explore random forest model we have prepared and saved in `model_rf.RDS`. The model is built with the following hyperparameter:

- set.seed(100) # the seed number   
- number = 5  # the number of k-fold cross validation   
- repeats = 3 # the number of the repeatation of iteration  

In your environment, load the random forest model under `model_rf` object using `readRDS()` function.

## 4.2 Random Forest Quiz
4. Which of the following are NOT among the advantages of random forest model?
  - [ ] Automatic feature selection  
  - [ ] It is a relatively fast model to compared to decision tree
  - [ ] Reduce bias in a model as it is the aggregate multiple decision trees  
  - [ ] It generates an unbiased estimate of out-of-box error  

Now check the summary of the final model we built using `model_rf$finalModel`.

5. From the summary of `model_rf$finalModel` above, the Out of Bag error rate from our model is 33.61%, what does this mean?
  - [ ] We have error 33.61% of our unseen data
  - [ ] We have error 33.61% of our train data
  - [ ] We have error 33.61% of our loan data
  - [ ] We have error 33.61% of our in-sample data

After built the model, let's predict the `data_train` and `data_test` based on `model_rf`, then please choose the `type = "raw"`.

## 4.3 Model evaluation

Next, let's evaluate the random forest we built using `confusionMatrix()`. Try to evaluate the performance of random forest model. How do you evaluate the model performance?

There are also other form of evaluation in random forest model, we can check which variable has a high significance to the prediction. We can check it using `varImp()` function and passed in to the `plot()` to get the visualization.

6. From the plot you have created, which is the most important variable?
  - [ ] checking_balance
  - [ ] months_loan_duration
  - [ ] amount
  - [ ] purpose

  
# 5 Naive Bayes

The last model we are trying to compare to is Naive Bayes model. There are several advantages in using this model, namely a few:  
- The model is relatively fast to train  
- It is estimating a probabilistic prediction  
- It can handle irrelevant features  

7. The following below are the characteristics of Naive Bayes, except...
  - [ ] Assume that among the predictor variables are independent  
  - [ ] Assume that between target and predictor variables are independent  
  - [ ] Skeweness due to data scarcity  
  
## 5.1 Naive Bayes model fitting

Now let's build a naive bayes model using `naiveBayes()` function from `e1071` package, then set the `laplace` parameter as 1. Store the model under `model_naive` before moving on to next section.

## 5.2 Predict the naive bayes model, using data train and data test

Using our test dataset we have created earlier, try to predict using `model_naive` and use `type="raw"`. The prediction will then results in a probabilities of positive class happening for each test dataset. Now, let us take a look to ROC and AUC performance, use the `prediction()` function from `ROCR` package to compare the positive class in `pred_naive` (`pred_naive[,2]`) with the actual data (`data_test$default`) and store it as `pred_roc` object.

Next, please use the performance function from ROCR package to help us define the axes and assign it to a `perf` object. To use the performance function, please define the arguments as below:
- prediction.obj = pred_roc
- measure = "tpr"
- x.measure = "fpr"

After you created `perf` object, plot the performance by passing it in `plot()` function. Try to evaluate the ROC Curve, see if there are any undesirable results from our model. Next, take a look at the auc using `performance()` function, and set the arguments `prediction.obj = pred_roc`, and `measure = "auc"` then save it under `auc` object.

8. From our naive bayes model above, how do you interpret the AUC value?
  - [ ] 78.44%, means the model performance is good because the closer to 1 the better
  - [ ] 78.44%, means the model performance is poor because the closer to 0 the better
  - [ ] 78.44%, the value of Area under ROC Curve did not give any information about model performance 

## 5.3 Model Evaluation Performance

Lastly, you can check the model performance for the naive bayes model using `confusionMatrix()` and compare it using the true label.

```
# your code here
```

# 6 Models Comparison

9. As a data scientist in financial institution, we are required to generate a rule-based model that can be implemented to the existing system. What is the best model for us to pick?
  - [ ] Naive Bayes because all the conditional probabilities is well calculated
  - [ ] Decision Tree because a decision tree model is easily translatable to a set of rules
  - [ ] Random Forest because it is possible to traceback the rule using variable importance information
  
10. Between all of the models, which model has better performance in term of identifying all high risk customers?
  - [ ] Naive Bayes
  - [ ] Decision Tree
  - [ ] Random Forest
