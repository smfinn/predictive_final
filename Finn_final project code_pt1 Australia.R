## Load in data ################################################################
au.train <- read.csv("~/Desktop/STAT 488 Predictive/Final Project/Australia/train.csv")
au.train <- au.train %>% mutate(rain_today = as.factor(rain_today),
                                rain_tomorrow = as.factor(rain_tomorrow),
                                location = as.factor(location),
                                wind_gust_dir = as.factor(wind_gust_dir),
                                wind_dir9am = as.factor(wind_dir9am),
                                wind_dir3pm = as.factor(wind_dir3pm),
                                month = as.numeric(substring(date, 6, 7)),
                                year = as.numeric(substring(date, 1, 4)))

au.test <- read.csv("~/Desktop/STAT 488 Predictive/Final Project/Australia/test.csv")
au.test <- au.test %>% mutate(rain_today = as.factor(rain_today),
                              location = as.factor(location),
                              wind_gust_dir = as.factor(wind_gust_dir),
                              wind_dir9am = as.factor(wind_dir9am),
                              wind_dir3pm = as.factor(wind_dir3pm),
                              month = as.numeric(substring(date, 6, 7)),
                              year = as.numeric(substring(date, 1, 4)))
au.sample <- read.csv("~/Desktop/STAT 488 Predictive/Final Project/Australia/sample_submission.csv")



## EDA #########################################################################
summary(au.train)

boxplot(au.train$min_temp)
boxplot(au.train$max_temp)
boxplot(au.train$wind_gust_dir)
boxplot(au.train$wind_gust_speed)
boxplot(au.train$wind_dir9am)
boxplot(au.train$wind_dir3pm)
boxplot(au.train$wind_speed9am)
boxplot(au.train$wind_speed3pm)
boxplot(au.train$humidity9am)
boxplot(au.train$humidity3pm)
boxplot(au.train$pressure9am)
boxplot(au.train$pressure3pm)
boxplot(au.train$cloud9am)
boxplot(au.train$cloud3pm)

hist(au.train$min_temp)
hist(au.train$max_temp)
hist(au.train$wind_gust_dir)
hist(au.train$wind_gust_speed)
hist(au.train$wind_dir9am)
hist(au.train$wind_dir3pm)
hist(au.train$wind_speed9am)
hist(au.train$wind_speed3pm)
hist(au.train$humidity9am)
hist(au.train$humidity3pm)
hist(au.train$pressure9am)
hist(au.train$pressure3pm)
hist(au.train$cloud9am)
hist(au.train$cloud3pm)

aggr(au.train[3:10])
aggr(au.train[11:17])
aggr(au.train[18:25])



## Imputation  #################################################################

# Impute data for training numeric variables
numeric.train <- au.train[, c(4,5,6,10,13,14,15,16,17,18,19,20,21,22,23,25,26)]

start <- Sys.time()
temp.numeric <- mice(numeric.train, method = "pmm", maxit = 3)
end <- Sys.time()
print(end - start)

summary(temp.numeric)
imputed.train <- complete(temp.numeric)

train.numsonly <- imputed.train
train.numsonly$rain_today <- as.numeric(train.numsonly$rain_today)
train.numsonly$rain_tomorrow <- au.train$rain_tomorrow
train.numsonly$id <- as.numeric(au.train$id)

write_csv(train.numsonly, "numeric Australia training dataset")

# Impute data for testing numeric variables
numeric.test <- au.test[, c(4,5,6,10,13,14,15,16,17,18,19,20,21,22,23,24,25)]
numeric.test$rain_today <- as.numeric(numeric.test$rain_today)

start <- Sys.time()
temp.numeric <- mice(numeric.test, method = "pmm", maxit = 3)
end <- Sys.time()
print(end - start)

summary(temp.numeric)
imputed.test <- complete(temp.numeric)

test.numsonly <- imputed.test
test.numsonly$id <- as.numeric(au.test$id)

write_csv(test.numsonly, "numeric Australia testing dataset")



## Logistic regression  ########################################################

au.logreg <- glm(rain_tomorrow ~ ., data = au.train[, -c(1,2,3,7,8)], 
                 family = "binomial")
summary(au.logreg)
logreg.preds <- predict(au.logreg, au.train)
logreg.preds <- as.factor(ifelse(logreg.preds > 0.5, 1, 0))
table(logreg.preds, au.train$rain_tomorrow)



## Reduce and fold training set ################################################

# Reduce and fold data for cross-validation
shuffle.train1 <- sample(1:34191, size = 1000)
shuffle.test1  <- sample(1:34191, size = 1000)
train1 <- train.numsonly[shuffle.train1,]
test1  <- train.numsonly[shuffle.test1,]

shuffle.train2 <- sample(1:34191, size = 1000)
shuffle.test2  <- sample(1:34191, size = 1000)
train2 <- train.numsonly[shuffle.train2,]
test2  <- train.numsonly[shuffle.test2,]

shuffle.train3 <- sample(1:34191, size = 1000)
shuffle.test3  <- sample(1:34191, size = 1000)
train3 <- train.numsonly[shuffle.train3,]
test3  <- train.numsonly[shuffle.test3,]

shuffle.train4 <- sample(1:34191, size = 1000)
shuffle.test4  <- sample(1:34191, size = 1000)
train4 <- train.numsonly[shuffle.train4,]
test4  <- train.numsonly[shuffle.test4,]

shuffle.train5 <- sample(1:34191, size = 1000)
shuffle.test5  <- sample(1:34191, size = 1000)
train5 <- train.numsonly[shuffle.train5,]
test5  <- train.numsonly[shuffle.test5,]

shuffle.train6 <- sample(1:34191, size = 1000)
shuffle.test6  <- sample(1:34191, size = 1000)
train6 <- train.numsonly[shuffle.train6,]
test6  <- train.numsonly[shuffle.test6,]

shuffle.train7 <- sample(1:34191, size = 1000)
shuffle.test7  <- sample(1:34191, size = 1000)
train7 <- train.numsonly[shuffle.train7,]
test7  <- train.numsonly[shuffle.test7,]

shuffle.train8 <- sample(1:34191, size = 1000)
shuffle.test8  <- sample(1:34191, size = 1000)
train8 <- train.numsonly[shuffle.train8,]
test8  <- train.numsonly[shuffle.test8,]

shuffle.train9 <- sample(1:34191, size = 1000)
shuffle.test9  <- sample(1:34191, size = 1000)
train9 <- train.numsonly[shuffle.train9,]
test9  <- train.numsonly[shuffle.test9,]

shuffle.train10 <- sample(1:34191, size = 1000)
shuffle.test10  <- sample(1:34191, size = 1000)
train10<- train.numsonly[shuffle.train10,]
test10 <- train.numsonly[shuffle.test10,]


## Train and tune random forest model ##########################################

# Basic out of the box random forest
rf1 <- randomForest(as.factor(rain_tomorrow) ~ ., data = train1)
varImpPlot(rf1)
summary(rf1$predicted)
rf1preds <- predict(rf1, newdata = test1)
table(rf1preds, train1$rain_tomorrow)

# Cross-validate for random forest characteristics
acc_vec <- rep(NA, 10)
findacc <- function(traindata, testdata) {
    forestmod <- randomForest(rain_tomorrow ~ ., data = traindata)
    testdata$preds <- predict(forestmod, newdata = testdata)
    accuracy <- sum(testdata$rain_tomorrow==testdata$preds)/nrow(testdata)
    return(accuracy)
}
acc_vec[1] <- findacc(train1, test1)
acc_vec[2] <- findacc(train2, test2)
acc_vec[3] <- findacc(train3, test3)
acc_vec[4] <- findacc(train4, test4)
acc_vec[5] <- findacc(train5, test5)
acc_vec[6] <- findacc(train6, test6)
acc_vec[7] <- findacc(train7, test7)
acc_vec[8] <- findacc(train8, test8)
acc_vec[9] <- findacc(train9, test9)
acc_vec[10]<- findacc(train10, test10)
mean(acc_vec)

# Cross-validate for number of trees
find_acc_ntree <- function(ntree) {
    acc_vec <- rep(NA, 10)
    findacc <- function(traindata, testdata) {
        forestmod <- randomForest(rain_tomorrow ~ ., data = traindata, ntree = ntree)
        testdata$preds <- predict(forestmod, newdata = testdata)
        accuracy <- sum(testdata$rain_tomorrow==testdata$preds)/nrow(testdata)
        return(accuracy)
    }
    acc_vec[1] <- findacc(train1, test1)
    acc_vec[2] <- findacc(train2, test2)
    acc_vec[3] <- findacc(train3, test3)
    acc_vec[4] <- findacc(train4, test4)
    acc_vec[5] <- findacc(train5, test5)
    acc_vec[6] <- findacc(train6, test6)
    acc_vec[7] <- findacc(train7, test7)
    acc_vec[8] <- findacc(train8, test8)
    acc_vec[9] <- findacc(train9, test9)
    acc_vec[10]<- findacc(train10, test10)
    return(mean(acc_vec))
}

ntree_accuracy <- NA
ntree_vec <- seq(100, 1000, length = 10)
for (i in 1:length(ntree_vec)){
    ntree_accuracy[i] <- find_acc_ntree(ntree_vec[i])
}
plot(ntree_vec, ntree_accuracy)
# Choose ntree = 300

# Cross-validate for node size
find_acc_node <- function(nodesize) {
    acc_vec <- rep(NA, 10)
    findacc <- function(traindata, testdata) {
        forestmod <- randomForest(rain_tomorrow ~ ., data = traindata, 
                                  ntree = 300, nodesize = nodesize)
        testdata$preds <- predict(forestmod, newdata = testdata)
        accuracy <- sum(testdata$rain_tomorrow==testdata$preds)/nrow(testdata)
        return(accuracy)
    }
    acc_vec[1] <- findacc(train1, test1)
    acc_vec[2] <- findacc(train2, test2)
    acc_vec[3] <- findacc(train3, test3)
    acc_vec[4] <- findacc(train4, test4)
    acc_vec[5] <- findacc(train5, test5)
    acc_vec[6] <- findacc(train6, test6)
    acc_vec[7] <- findacc(train7, test7)
    acc_vec[8] <- findacc(train8, test8)
    acc_vec[9] <- findacc(train9, test9)
    acc_vec[10]<- findacc(train10, test10)
    return(mean(acc_vec))
}

nodes_accuracy <- NA
nodes_vec <- seq(1, 10, length = 10)
for (i in 1:length(nodes_vec)){
    nodes_accuracy[i] <- find_acc_node(nodes_vec[i])
}
plot(nodes_vec, nodes_accuracy)
# Choose nodesize = 1

# Cross-validate for number of variables to try at each split
find_acc_mtry <- function(mtry) {
    acc_vec <- rep(NA, 10)
    findacc <- function(traindata, testdata) {
        forestmod <- randomForest(rain_tomorrow ~ ., data = traindata, 
                                  ntree = 300, nodesize = 1, mtry = mtry)
        testdata$preds <- predict(forestmod, newdata = testdata)
        accuracy <- sum(testdata$rain_tomorrow==testdata$preds)/nrow(testdata)
        return(accuracy)
    }
    acc_vec[1] <- findacc(train1, test1)
    acc_vec[2] <- findacc(train2, test2)
    acc_vec[3] <- findacc(train3, test3)
    acc_vec[4] <- findacc(train4, test4)
    acc_vec[5] <- findacc(train5, test5)
    acc_vec[6] <- findacc(train6, test6)
    acc_vec[7] <- findacc(train7, test7)
    acc_vec[8] <- findacc(train8, test8)
    acc_vec[9] <- findacc(train9, test9)
    acc_vec[10]<- findacc(train10, test10)
    return(mean(acc_vec))
}

mtry_accuracy <- NA
mtry_vec <- seq(1, 10, length = 10)
for (i in 1:length(mtry_vec)){
    mtry_accuracy[i] <- find_acc_mtry(mtry_vec[i])
}
plot(mtry_vec, mtry_accuracy)
# Choose mtry = 4

# With tuned parameters
acc_vec <- rep(NA, 10)
findacc <- function(traindata, testdata) {
    forestmod <- randomForest(rain_tomorrow ~ ., data = traindata,
                              ntree = 300, nodesize = 5, mtry = 4)
    testdata$preds <- predict(forestmod, newdata = testdata)
    accuracy <- sum(testdata$rain_tomorrow==testdata$preds)/nrow(testdata)
    return(accuracy)
}
acc_vec[1] <- findacc(train1, test1)
acc_vec[2] <- findacc(train2, test2)
acc_vec[3] <- findacc(train3, test3)
acc_vec[4] <- findacc(train4, test4)
acc_vec[5] <- findacc(train5, test5)
acc_vec[6] <- findacc(train6, test6)
acc_vec[7] <- findacc(train7, test7)
acc_vec[8] <- findacc(train8, test8)
acc_vec[9] <- findacc(train9, test9)
acc_vec[10]<- findacc(train10, test10)
mean(acc_vec)


## Make final predictions ######################################################

train.numsonly$rain_tomorrow <- ifelse(train.numsonly$rain_tomorrow == "0", 0, 1)

start <- Sys.time()
finalmod <- randomForest(as.numeric(rain_tomorrow) ~ ., data = train.numsonly, 
                         ntree = 300, nodesize = 5, mtry = 4)
end <- Sys.time()
print(end - start)

final.preds <- predict(finalmod, newdata = test.numsonly)
summary(final.preds)

preds <- data.frame(id = au.test$id,
                    rain_tomorrow = final.preds)
write_csv(preds, "Australia rain tomorrow predictions finally something.csv")
