Congratulation! This is the end of Classification in Machine Learning 2. The last part of this course is closed by filling this quiz. In this quiz, you will use the loan dataset. The data is stored as csv file in this repository as loan.csv file. The dataset is downloaded from [UCI Machine Learning](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)). 

To accomplish this assignment, you need to build your classification model using naive bayes, decision tree and random forest algorithms by following these steps:

# 1 Data Exploration

Please load and investigate the data given `loan.csv` and assign it named `loan`, then investigate the data using `str()` or `glimpse()` function.

```
# your code here
```

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

For example, our boss wants to know the number of default or not for each purpose of loan. So, let us do some data aggregation to get the answer.

```
# your code
```

1. Based on the exploration above, what is the highest number of purpose to not default?
  - [] furniture/applicances
  - [] car
  - [] business
  - [] education

## 1.1 Know your business question

As a data scientist who will help the analyst took the decision, first thing that should we do is know what kind of problem we want to solve. Based on the data and its explanation, let us identify the bank loans that are likely to default, hence we will predict if the default variable in the data will be `yes` or `no`, then the target variable that will be used is `default`.

# 2 Cross Validation

Before we build our model, we should split the dataset into train and test data. In this step, please split the data into 80% train and 20% test proportion use `sample()`function, `set.seed(100)`, and store it as `data_train` and `data_test`

```
# your code here
```
## 2.1 Check the proportion of our target variable

In this phase, please check the target variable on `data_train$default` proportion using `table()` or `prop.table()` function

```
# your code here
```

Based on the proportion of the target variable above, we can conclude that our target variable is not balance, hence we should balance it first before do the cross validation. Please balance the proportion of the target variable on `data_train` using `downSample()` function from `caret` package.

```
# your code here
```
# 3 Decision Tree

## 3.1 Decision Tree Model Fitting

After we have splitted our data into `data_train` and `data_test`, let's build our first model of decision tree using `ctree()` function and store it as `model_dt` object. To tune our model, please set the `mincriterion = 0.10`. 
```
# your code here
```

## 3.2 Decision Tree Quiz

2. In our decision model, the goals of setting the mincriterion = 0.10 is to ...
  - [] To do the pruning, we let the tree which has maximum p-value <= 0.10 to split the node
  - [] To do the pruning, we let the tree which has maximum p-value <= 0.90 to split the node
  - [] To do the pruning, we let the tree which has maximum 10% data in the terminal nodes
  - [] To do the pruning, we let the tree which has maximum 10% data in the terminal nodes

To get a picture of our model, please plot the model, and use the `type = "simple"`

```
# your code here
```

## 3.3 Predict on data train and test

To get our prediction, please predict the model towards `data_train` and `data_test` based on `model_dt` using `predict()` function and choose the `type = "response"`

```
# pred_train_dt <-
# pred_test_dt <-
```

# 3.4 Model Evaluation

The last part of build the model is having to know whether our model is quite good or not, so we should evaluate our model performance. Please use the `confusionMatrix()` to get several model performance. Then, compare the result using `data_train` and `data_test`.

```
# your code here
```
3. From the decision tree performance above, we can conclude that our decision tree model is ...
  - [] Quite good
  - [] Overfitting
  - [] Underfitting


# 4 Random Forest
## 4.1 Random forest model fitting
Now, given our second model using another algorithm, that is random forest with these specifications :    

- set.seed(100) # the seed number   
- number = 5  # the number of k-fold cross validation   
- repeats = 3 # the number of the repeatation of iteration  

The model has been saved in `model_rf.RDS`. Please just load the .RDS file in the chunk below using `readRDS()` function.

```
model_rf <- 
```

## 4.2 Random Forest Quiz
4. Which of the following are among the advantages of a random forest? Select all that apply. 
  - []  Automatic feature selection 
  - []  Reduce bias in a model as it is the aggregate of many decision trees, hence a significantly lower risk of overfitting compared to a decision tree 
  - []  It generates an internal unbiased estimate of out-of-box error 

Check the summary of the final model we built using `model_rf$finalModel`

```
# your code here
```
5. From the summary of `model_rf$finalModel` above, the Out of Bag error rate from our model is 33.61%, what does this mean?
  - [] We have error 33.61% for our unseen data
  - [] We have error 33.61% for our train data
  - [] We have error 33.61% for our loan data

After built the model, let's predict the `data_train` and `data_test` based on `model_rf`, then please choose the `type = "raw"`

```
# your code here
```

## 4.3 Model evaluation

Next, let's evaluate the random forest we built using `confusionMatrix()`

```
# your code here
```

## 4.4 Variable importance

In random forest, we can check what is the variable influence the model most. We can check it using `varImp()` function

```
# your code here
```

6. From the output above, What is the most important variable based on our model?
  - [] checking_balance
  - [] months_loan_duration
  - [] amount

  
# 5 Naive Bayes

## 5.1 Naive Bayes Quiz

7. The following below are the characteristics of Naive Bayes, except
  - [] Assume that among the predictor variables are independent 
  - [] Assume that among the target and predictor variables are independent
  - [] Skweness due to data scarcity
  
## 5.2 Naive Bayes model fitting

Finally, we arrived at the last model that is naive bayes. Build a naive bayes model using `naiveBayes()` function from `e1071` package, then set the `laplace = 1`.

```
# your code here
```

## 5.3 Predict the naive bayes model, using data train and data test

After we built the naive bayes model, then we can predict the `data_test` based on `model_naive` and use `type = "raw"`

```
# your code here
```

Now, let us take a look to ROC and AUC performance, use the `prediction()` function from `ROCR` package to compare the positive class in `pred_naive` (`pred_naive[,2]`) with the actual data (`data_test$default`) and store it as `pred_roc` object.

Next, please use the performance function from ROCR package to help us define the axes and assign it to a `perf` object. To use the performance function, please define the arguments as below:
- prediction.obj = pred_roc
- measure = "tpr"
- x.measure = "fpr"

```
perf <- 
```

After that, plot the performance using `plot()` function

```
# your code here
```

The next step is to get the AUC (Area Under Curve) value. Take a look the auc using `performance()` function, and set the arguments `prediction.obj = pred_roc`, and `measure = "auc"` and saved it  `auc`. object

```
auc <-
```

8. From the model above, how is the value of Area Under the ROC Curve?
  - [] 78.44%, means the model performance is quite good because the closer to 1 the better
  - [] 78.44%, means the model performance is quite bad because the closer to 1 the worse
  - [] 78.44%, the value of Area Uunder ROC Curve did not give any information about model performance 

## 5.4 Model Evaluation Performance

Then, check the model evaluation performance for the naive bayes model using `confusionMatrix()`

```
# your code here
```

# 6 Comparing The Model

9. As a data scientist in a bank, our bosses wants to know the interpretation behind the model. What is the best model should we choose to present?
  - [] Naive Bayes because we can intrepret the probability
  - [] Decision Tree because we can know what is the most important variable by seeing what is the root
  - [] Random Forest because we can know the most important variable to our target variable
  
10. From the three models we have built, if we, as a data scientist want to take as much as the number of people who are likely to default, what is the best model to choose?
  - [] Naive Bayes
  - [] Decision Tree
  - [] Random Forest