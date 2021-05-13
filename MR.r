######################################Problem 1#########################
# Multinomial Logit Model
install.packages("mlogit")
require('mlogit')
install.packages("nnet")
require('nnet')
library(readr)

#loading dataset
mdata <- read_csv("C:\\Users\\hp\\Desktop\\Multinomial R assi\\mdata.csv")

#dropping unwanted columns
mdata <- mdata[,-c(1,2)]
head(mdata)
tail(mdata)
summary(mdata)

#getting count of response variable
table(mdata$prog) 

# Data Partitioning
n <-  nrow(mdata)
n1 <-  n * 0.85
n2 <-  n - n1
train_index <- sample(1:n, n1)
train <- mdata[train_index, ]
test <-  mdata[-train_index, ]

commute <- multinom(prog ~ female + ses + schtyp + read + write + math + science + honors, data = train)
summary(commute)

#Significance of Regression Coefficients
z <- summary(commute)$coefficients / summary(commute)$standard.errors

p_value <- (1 - pnorm(abs(z), 0, 1)) * 2

summary(commute)$coefficients
p_value

# odds ratio 
exp(coef(commute))

# check for fitted values on training data
prob <- fitted(commute)

# Predicted on test data
pred_test <- predict(commute, newdata =  test, type = "probs") # type="probs" is to calculate probabilities
pred_test

# Find the accuracy of the model
class(pred_test)
pred_test <- data.frame(pred_test)
View(pred_test)
pred_test["prediction"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

predtest_name <- apply(pred_test, 1, get_names)
pred_test$prediction <- predtest_name
View(pred_test)

# Confusion matrix
table(predtest_name, test$prog)

# confusion matrix visualization
barplot(table(predtest_name, test$prog), beside = T, col =c("red", "lightgreen", "blue", "orange"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")

# Accuracy on test data
mean(predtest_name == test$prog)

# Training Data
pred_train <- predict(commute, newdata =  train, type="probs") # type="probs" is to calculate probabilities
pred_train

# Find the accuracy of the model
class(pred_train)
pred_train <- data.frame(pred_train)
View(pred_train)
pred_train["prediction"] <- NULL

predtrain_name <- apply(pred_train, 1, get_names)
pred_train$prediction <- predtrain_name
View(pred_train)

# Confusion matrix
table(predtrain_name, train$prog)

# confusion matrix visualization
barplot(table(predtrain_name, train$prog), beside = T, col =c("red", "lightgreen", "blue", "orange"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")

# Accuracy 
mean(predtrain_name == train$prog)

#######################################Problem 2###########################################
# Multinomial Logit Model
#install.packages("mlogit")
require('mlogit')
#install.packages("nnet")
require('nnet')
library(readr)

#loading dataset
loan <- read_csv("C:\\Users\\hp\\Desktop\\Multinomial R assi\\loan.csv")

#subsetting the csv file
loan <- loan[,1:17]

#dropping unwanted columns
loan <- loan[,-c(1,2,11,12,16)]

#converting 'init_rate' to float (removing %)
loan$int_rate <- sapply(loan$int_rate, gsub , pattern = '%' , replacement = '')

head(loan)
tail(loan)
names(loan)

#getting count of response variable
table(loan$loan_status)

# Data Partitioning
loan <- loan[1:1000,]
n <-  nrow(loan)
n1 <-  n * 0.85
n2 <-  n - n1
train_index <- sample(1:n, n1)
train <- loan[train_index, ]
test <-  loan[-train_index, ]

commute <- multinom(loan_status ~ ., data = train)
summary(commute)
#loan_amnt + funded_amnt + funded_amnt_inv + term + int_rate + installment + grade + sub_grade + home_ownership + annual_inc + verification_status
#Significance of Regression Coefficients
z <- summary(commute)$coefficients / summary(commute)$standard.errors

p_value <- (1 - pnorm(abs(z), 0, 1)) * 2

summary(commute)$coefficients
p_value

# odds ratio 
exp(coef(commute))

# check for fitted values on training data
prob <- fitted(commute)

# Predicted on test data
pred_test <- predict(commute, newdata =  test, type = "probs") # type="probs" is to calculate probabilities
pred_test

# Find the accuracy of the model
class(pred_test)
pred_test <- data.frame(pred_test)
View(pred_test)
pred_test["prediction"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

predtest_name <- apply(pred_test, 1, get_names)
pred_test$prediction <- predtest_name
View(pred_test)

# Confusion matrix
table(predtest_name, test$loan_status)

# confusion matrix visualization
barplot(table(predtest_name, test$loan_status), beside = T, col =c("red", "lightgreen", "blue", "orange"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")

# Accuracy on test data
mean(predtest_name == test$loan_status)

# Training Data
pred_train <- predict(commute, newdata =  train, type="probs") # type="probs" is to calculate probabilities
pred_train

# Find the accuracy of the model
class(pred_train)
pred_train <- data.frame(pred_train)
View(pred_train)
pred_train["prediction"] <- NULL

predtrain_name <- apply(pred_train, 1, get_names)
pred_train$prediction <- predtrain_name
View(pred_train)

# Confusion matrix
table(predtrain_name, train$loan_status)

# confusion matrix visualization
barplot(table(predtrain_name, train$loan_status), beside = T, col =c("red", "lightgreen", "blue", "orange"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")

# Accuracy 
mean(predtrain_name == train$loan_status)

######################################END###########################################