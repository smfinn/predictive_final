library(tidyverse)
library(randomForest)
library(mice)
library(stringr)
library(class)
library(VIM)
library(xgboost)
library(ggcorrplot)
library(lubridate)
library(SentimentAnalysis)
library(tm)
library(gam)
library(VGAM)

################################################################################
## Cleaning and feature preparation ############################################
bnb.train <- read.csv("~/Desktop/STAT 488 Predictive/Final Project/AirBnB/train.csv")
bnb.test <- read.csv("~/Desktop/STAT 488 Predictive/Final Project/AirBnB/test.csv")
bnb.sample <- read.csv("~/Desktop/STAT 488 Predictive/Final Project/AirBnB/sample_submission.csv")

# Calculate length of listing name
bnb.train$namelength <- str_count(bnb.train$name, "\\w+")
bnb.test$namelength <- str_count(bnb.test$name, "\\w+")

# Calculate days since last review
bnb.train$last_review <- as_date(bnb.train$last_review)
bnb.test$last_review <- as_date(bnb.test$last_review)
bnb.train$days.lastreview <- as.numeric(as_date("2021-12-14") - bnb.train$last_review)
bnb.test$days.lastreview <- as.numeric(as_date("2021-12-14") - bnb.test$last_review)
bnb.train$days.lastreview <- bnb.train$days.lastreview - min(bnb.train$days.lastreview, na.rm = T)
bnb.test$days.lastreview <- bnb.test$days.lastreview - min(bnb.test$days.lastreview, na.rm = T)
bnb.train$days.lastreview[is.na(bnb.train$days.lastreview)] <- 0
bnb.test$days.lastreview[is.na(bnb.test$days.lastreview)] <- 0

# Replace NAs with 0s
bnb.train$reviews_per_month[is.na(bnb.train$reviews_per_month)] <- 0
bnb.train$namelength[is.na(bnb.train$namelength)] <- 0
bnb.test$reviews_per_month[is.na(bnb.test$reviews_per_month)] <- 0
bnb.test$namelength[is.na(bnb.test$namelength)] <- 0

# Make new months listed variable
bnb.train$months.listed <- ifelse(bnb.train$number_of_reviews > 0, 
                                  bnb.train$number_of_reviews / bnb.train$reviews_per_month,
                                  0)
bnb.test$months.listed <- ifelse(bnb.test$number_of_reviews > 0, 
                                 bnb.test$number_of_reviews / bnb.test$reviews_per_month,
                                 0)

# Make monthly rental indicator variable
bnb.train$rental <- ifelse(bnb.train$minimum_nights >= 30, 1, 0)
bnb.test$rental <- ifelse(bnb.test$minimum_nights >= 30, 1, 0)

# Encode neighborhood groups
bnb.train$neighbourhood_num <- ifelse(bnb.train$neighbourhood_group == "Manhattan", 1,
                                      ifelse(bnb.train$neighbourhood_group == "Brooklyn", 2, 
                                             ifelse(bnb.train$neighbourhood_group == "Bronx", 3,
                                                    ifelse(bnb.train$neighbourhood_group == "Queens", 4, 
                                                           ifelse(bnb.train$neighbourhood_group == "Staten Island", 5, 0)))))
bnb.test$neighbourhood_num <- ifelse(bnb.test$neighbourhood_group == "Manhattan", 1,
                                     ifelse(bnb.test$neighbourhood_group == "Brooklyn", 2, 
                                            ifelse(bnb.test$neighbourhood_group == "Bronx", 3,
                                                   ifelse(bnb.test$neighbourhood_group == "Queens", 4, 
                                                          ifelse(bnb.test$neighbourhood_group == "Staten Island", 5, 0)))))

# Encode room types
bnb.train$room_type_num <- ifelse(bnb.train$room_type == "Entire home/apt", 1,
                                  ifelse(bnb.train$room_type == "Private room", 2, 
                                         ifelse(bnb.train$room_type == "Shared room", 3, 0)))
bnb.test$room_type_num <- ifelse(bnb.test$room_type == "Entire home/apt", 1,
                                 ifelse(bnb.test$room_type == "Private room", 2, 
                                        ifelse(bnb.test$room_type == "Shared room", 3, 0)))


################################################################################
## EDA #########################################################################
hist(bnb.train$price)
hist(bnb.train$price[bnb.train$price < 1000])

hist(bnb.train$latitude)
hist(bnb.train$longitude)
hist(bnb.train$minimum_nights)
hist(bnb.train$minimum_nights[bnb.train$minimum_nights<30])
hist(bnb.train$number_of_reviews)
hist(bnb.train$reviews_per_month)
summary(bnb.train$reviews_per_month)
hist(bnb.train$calculated_host_listings_count)
summary(bnb.train$calculated_host_listings_count)
hist(bnb.train$availability_365)
hist(bnb.train$namelength)
hist(bnb.train$months.listed)
hist(bnb.train$days.lastreview)

table(bnb.train$rental)
table(bnb.train$neighbourhood_group)
table(bnb.train$room_type)

correlations <- round(cor(bnb.train[, c(7, 8, 10:12, 14:17, 19)], use = "all.obs"), 3)
p.mat <- cor_pmat(bnb.train[, c(7, 8, 10:12, 14:17, 19)])
ggcorrplot(correlations, type = "upper", hc.order = TRUE, p.mat = p.mat)


################################################################################
## Sentiment analysis ##########################################################

bnb.train$name <- as.vector(bnb.train$name)
sentiment <- analyzeSentiment(bnb.train$name)
sentimentGI <- sentiment$SentimentGI
bnb.train <- cbind(bnb.train, sentimentGI)
bnb.train$sentimentGI[is.na(bnb.train$sentimentGI)] <- 0
write_csv(bnb.train, "AirBnB training set with sentiment analysis scores.csv")

bnb.test$name <- as.vector(bnb.test$name)
sentiment <- analyzeSentiment(bnb.test$name)
sentimentGI <- sentiment$SentimentGI
bnb.test <- cbind(bnb.test, sentimentGI)
bnb.test$sentimentGI[is.na(bnb.test$sentimentGI)] <- 0
write_csv(bnb.test, "AirBnB testing set with sentiment analysis scores.csv")


################################################################################
## Reduce and fold data ########################################################
red.bnb.train <- bnb.train[, c(7:8, 10:12, 14:23)]
red.bnb.test <- bnb.test[, c(7:8, 10:11, 13:22)]

# Reduce and fold data for cross-validation
shuffle.train1 <- sample(1:34226, size = 1000)
shuffle.test1  <- sample(1:34226, size = 1000)
train1 <- red.bnb.train[shuffle.train1,]
test1  <- red.bnb.train[shuffle.test1,]

shuffle.train2 <- sample(1:34226, size = 1000)
shuffle.test2  <- sample(1:34226, size = 1000)
train2 <- red.bnb.train[shuffle.train2,]
test2  <- red.bnb.train[shuffle.test2,]

shuffle.train3 <- sample(1:34226, size = 1000)
shuffle.test3  <- sample(1:34226, size = 1000)
train3 <- red.bnb.train[shuffle.train3,]
test3  <- red.bnb.train[shuffle.test3,]

shuffle.train4 <- sample(1:34226, size = 1000)
shuffle.test4  <- sample(1:34226, size = 1000)
train4 <- red.bnb.train[shuffle.train4,]
test4  <- red.bnb.train[shuffle.test4,]

shuffle.train5 <- sample(1:34226, size = 1000)
shuffle.test5  <- sample(1:34226, size = 1000)
train5 <- red.bnb.train[shuffle.train5,]
test5  <- red.bnb.train[shuffle.test5,]

shuffle.train6 <- sample(1:34226, size = 1000)
shuffle.test6  <- sample(1:34226, size = 1000)
train6 <- red.bnb.train[shuffle.train6,]
test6  <- red.bnb.train[shuffle.test6,]

shuffle.train7 <- sample(1:34226, size = 1000)
shuffle.test7  <- sample(1:34226, size = 1000)
train7 <- red.bnb.train[shuffle.train7,]
test7  <- red.bnb.train[shuffle.test7,]

shuffle.train8 <- sample(1:34226, size = 1000)
shuffle.test8  <- sample(1:34226, size = 1000)
train8 <- red.bnb.train[shuffle.train8,]
test8  <- red.bnb.train[shuffle.test8,]

shuffle.train9 <- sample(1:34226, size = 1000)
shuffle.test9  <- sample(1:34226, size = 1000)
train9 <- red.bnb.train[shuffle.train9,]
test9  <- red.bnb.train[shuffle.test9,]

shuffle.train10 <- sample(1:34226, size = 1000)
shuffle.test10  <- sample(1:34226, size = 1000)
train10<- red.bnb.train[shuffle.train10,]
test10 <- red.bnb.train[shuffle.test10,]

train.sets <- list(train1, train2, train3, train4, train5,
                   train6, train7, train8, train9, train10)
test.sets <- list(test1, test2, test3, test4, test5,
                  test6, test7, test8, test9, test10)


################################################################################
## Build and cross-validate GAM ################################################
gam1 <- gam(price ~ ., data = red.bnb.train)
(rmse1 <- mean((gam1$fitted.values - red.bnb.train$price)^2))

gam2 <- gam(price ~ s(latitude) + s(longitude) + minimum_nights + number_of_reviews + 
                reviews_per_month + calculated_host_listings_count + availability_365 + 
                namelength + days.lastreview + months.listed + rental + sentimentGI +
                neighbourhood_num + room_type_num, data = red.bnb.train)
(rmse2 <- mean((gam2$fitted.values - red.bnb.train$price)^2))

gam3 <- gam(price ~ s(latitude) + s(longitude) + minimum_nights + 
                lo(number_of_reviews) + reviews_per_month + calculated_host_listings_count + availability_365 + 
                namelength + days.lastreview + months.listed + rental + sentimentGI +
                neighbourhood_num + room_type_num, data = red.bnb.train)
(rmse3 <- mean((gam3$fitted.values - red.bnb.train$price)^2))

gam4 <- gam(price ~ s(latitude) + s(longitude) + minimum_nights + 
                s(number_of_reviews) + s(reviews_per_month) + 
                calculated_host_listings_count + availability_365 + 
                namelength + days.lastreview + months.listed + rental + sentimentGI +
                neighbourhood_num + room_type_num, data = red.bnb.train)
(rmse4 <- mean((gam4$fitted.values - red.bnb.train$price)^2))

gam5 <- gam(price ~ s(latitude) + s(longitude) + minimum_nights + 
                s(number_of_reviews) + s(reviews_per_month) + 
                lo(calculated_host_listings_count) + availability_365 + 
                namelength + days.lastreview + months.listed + rental + sentimentGI +
                neighbourhood_num + room_type_num, data = red.bnb.train)
(rmse5 <- mean((gam5$fitted.values - red.bnb.train$price)^2))

gam6 <- gam(price ~ s(latitude) + s(longitude) + minimum_nights + 
                s(number_of_reviews) + s(reviews_per_month) + 
                lo(calculated_host_listings_count) + s(availability_365) + 
                namelength + days.lastreview + months.listed + rental + sentimentGI +
                neighbourhood_num + room_type_num, data = red.bnb.train)
(rmse6 <- mean((gam6$fitted.values - red.bnb.train$price)^2))

gam7 <- gam(price ~ s(latitude) + s(longitude) + minimum_nights + 
                s(number_of_reviews) + s(reviews_per_month) + 
                lo(calculated_host_listings_count) + s(availability_365) + 
                s(namelength) + days.lastreview + months.listed + rental + sentimentGI +
                neighbourhood_num + room_type_num, data = red.bnb.train)
(rmse7 <- mean((gam7$fitted.values - red.bnb.train$price)^2))

gam8 <- gam(price ~ s(latitude) + s(longitude) + minimum_nights + 
                s(number_of_reviews) + s(reviews_per_month) + 
                lo(calculated_host_listings_count) + s(availability_365) + 
                s(namelength) + s(days.lastreview) + months.listed + rental + sentimentGI +
                neighbourhood_num + room_type_num, data = red.bnb.train)
(rmse8 <- mean((gam8$fitted.values - red.bnb.train$price)^2))

gam9 <- gam(price ~ s(latitude) + s(longitude) + minimum_nights + 
                s(number_of_reviews) + s(reviews_per_month) + 
                lo(calculated_host_listings_count) + s(availability_365) + 
                s(namelength) + s(days.lastreview) + s(months.listed) + rental + 
                sentimentGI + neighbourhood_num + room_type_num, 
            data = red.bnb.train)
(rmse9 <- mean((gam9$fitted.values - red.bnb.train$price)^2))

gam10 <- gam(price ~ s(latitude) + s(longitude) + minimum_nights + 
                 s(number_of_reviews) + s(reviews_per_month) + 
                 lo(calculated_host_listings_count) + s(availability_365) + 
                 s(namelength) + s(days.lastreview) + s(months.listed) + rental + 
                 lo(sentimentGI) + neighbourhood_num + room_type_num, 
             data = red.bnb.train)
(rmse10 <- mean((gam10$fitted.values - red.bnb.train$price)^2))

anova(gam1, gam10)

mse.spline <- c()
mse.mod <- c()
splines <- seq(2, 10)
for(i in 1:length(splines)) {
    for(j in 1:10) {
        gam <- gam(price ~ s(latitude, df = 3) + s(longitude, df = 3) + minimum_nights + 
                       s(number_of_reviews, df = 5) + s(reviews_per_month, df = 7) + 
                       lo(calculated_host_listings_count) + s(availability_365, df = 7) + 
                       s(namelength, df = 2) + s(days.lastreview, df = 2) + 
                       s(months.listed, df = splines[i]) + rental + 
                       lo(sentimentGI) + neighbourhood_num + room_type_num, 
                   data = train.sets[[j]])
        mse.mod[j] <- mean((predict(gam, test.sets[[j]]) - test.sets[[j]]$price)^2)
    }
    mse.spline[i] <- mean(mse.mod)
}
plot(splines, mse.spline)
best.spline <- splines[which.min(mse.spline)]
# Best spline df for latitude: 3
# Best spline df for longitude: 3
# Best spline df for number of reviews: 5
# Best spline df for reviews per month: 7
# Best spline df for availability: 7
# Best spline df for name length: 2
# Best spline df for days since last review: 2
# Best spline df for months listed: 3

mse.loess <- c()
spans <- seq(0.25, 0.75, 0.05)
for(i in 1:length(spans)) {
    for(j in 1:10) {
        gam <- gam(price ~ s(latitude, df = 3) + s(longitude, df = 3) + minimum_nights + 
                       s(number_of_reviews, df = 5) + s(reviews_per_month, df = 7) + 
                       lo(calculated_host_listings_count, span = 0.25) + s(availability_365, df = 7) + 
                       s(namelength, df = 2) + s(days.lastreview, df = 2) + 
                       s(months.listed, df = 3) + rental + 
                       lo(sentimentGI, span = spans[i]) + neighbourhood_num + room_type_num, 
                   data = train.sets[[j]])
        mse.mod[j] <- mean((predict(gam, test.sets[[j]]) - test.sets[[j]]$price)^2)
    }
    mse.loess[i] <- mean(mse.mod)
}
plot(spans, mse.loess)
best.loess <- spans[which.min(mse.loess)]
# Best loess span for host listings count: 0.25
# Best loess span for listing name sentiment score: 0.5


################################################################################
## Finalize GAM and make predictions ###########################################
gam.final <- gam(price ~ s(latitude, df = 3) + s(longitude, df = 3) + 
                     minimum_nights + s(number_of_reviews, df = 5) + 
                     s(reviews_per_month, df = 7) + 
                     lo(calculated_host_listings_count, span = 0.25) + 
                     s(availability_365, df = 7) + s(namelength, df = 2) + 
                     s(days.lastreview, df = 2) + s(months.listed, df = 3) + rental + 
                     lo(sentimentGI, span = 0.5) + neighbourhood_num + 
                     room_type_num, data = red.bnb.train)
(rmse.fin <- mean((gam.final$fitted.values - red.bnb.train$price)^2))
anova(gam.final, gam10)

gam.preds <- predict(gam.final, red.bnb.test)
preds1 <- data.frame(id = bnb.test$id, 
                     price = gam.preds)

write_csv(preds1, "AirBnB price prediction using GAM.csv")


################################################################################
## Build and cross-validate random forest ######################################
# Basic out of the box random forest
rf1 <- randomForest(price ~ ., data = red.bnb.train)
varImpPlot(rf1)
summary(rf1$predicted)
rf1preds <- predict(rf1, newdata = red.bnb.test)
preds2 <- data.frame(id = bnb.test$id,
                     price = rf1preds)
write_csv(preds2, "AirBnB price predictions using out of box random forest.csv")

# Cross-validate for number of trees
find_mse_ntree <- function(ntree) {
    mse_vec <- rep(NA, 10)
    find_mse <- function(traindata, testdata) {
        forestmod <- randomForest(price ~ ., data = traindata, ntree = ntree)
        mse <- mean((predict(forestmod, newdata = testdata) - testdata$price)^2)
        return(mse)
    }
    mse_vec[1] <- find_mse(train1, test1)
    mse_vec[2] <- find_mse(train2, test2)
    mse_vec[3] <- find_mse(train3, test3)
    mse_vec[4] <- find_mse(train4, test4)
    mse_vec[5] <- find_mse(train5, test5)
    mse_vec[6] <- find_mse(train6, test6)
    mse_vec[7] <- find_mse(train7, test7)
    mse_vec[8] <- find_mse(train8, test8)
    mse_vec[9] <- find_mse(train9, test9)
    mse_vec[10]<- find_mse(train10, test10)
    return(mean(mse_vec))
}

ntree_accuracy <- NA
ntree_vec <- seq(100, 1000, length = 10)
for (i in 1:length(ntree_vec)){
    ntree_accuracy[i] <- find_mse_ntree(ntree_vec[i])
}
plot(ntree_vec, ntree_accuracy)
# Choose ntree = 500

# Cross-validate for node size
find_mse_node <- function(nodesize) {
    mse_vec <- rep(NA, 10)
    find_mse <- function(traindata, testdata) {
        forestmod <- randomForest(price ~ ., data = traindata, 
                                  ntree = 500, nodesize = nodesize)
        mse <- mean((predict(forestmod, newdata = testdata) - testdata$price)^2)
        return(mse)
    }
    mse_vec[1] <- find_mse(train1, test1)
    mse_vec[2] <- find_mse(train2, test2)
    mse_vec[3] <- find_mse(train3, test3)
    mse_vec[4] <- find_mse(train4, test4)
    mse_vec[5] <- find_mse(train5, test5)
    mse_vec[6] <- find_mse(train6, test6)
    mse_vec[7] <- find_mse(train7, test7)
    mse_vec[8] <- find_mse(train8, test8)
    mse_vec[9] <- find_mse(train9, test9)
    mse_vec[10]<- find_mse(train10, test10)
    return(mean(mse_vec))
}

nodes_accuracy <- NA
nodes_vec <- seq(1, 10, length = 10)
for (i in 1:length(nodes_vec)){
    nodes_accuracy[i] <- find_mse_node(nodes_vec[i])
}
plot(nodes_vec, nodes_accuracy)
# Choose nodesize = 6

# Cross-validate for number of variables to try at each split
find_mse_mtry <- function(mtry) {
    mse_vec <- rep(NA, 10)
    find_mse <- function(traindata, testdata) {
        forestmod <- randomForest(price ~ ., data = traindata, 
                                  ntree = 500, nodesize = 6, mtry = mtry)
        mse <- mean((predict(forestmod, newdata = testdata) - testdata$price)^2)
        return(mse)
    }
    mse_vec[1] <- find_mse(train1, test1)
    mse_vec[2] <- find_mse(train2, test2)
    mse_vec[3] <- find_mse(train3, test3)
    mse_vec[4] <- find_mse(train4, test4)
    mse_vec[5] <- find_mse(train5, test5)
    mse_vec[6] <- find_mse(train6, test6)
    mse_vec[7] <- find_mse(train7, test7)
    mse_vec[8] <- find_mse(train8, test8)
    mse_vec[9] <- find_mse(train9, test9)
    mse_vec[10]<- find_mse(train10, test10)
    return(mean(mse_vec))
}

mtry_accuracy <- NA
mtry_vec <- seq(1, 10, length = 10)
for (i in 1:length(mtry_vec)){
    mtry_accuracy[i] <- find_mse_mtry(mtry_vec[i])
}
plot(mtry_vec, mtry_accuracy)
# Choose mtry = 3

# With tuned parameters
rf.tuned <- randomForest(price ~ ., data = red.bnb.train, ntree = 600,
                         nodesize = 6, mtry = 3)
mean((rf.tuned$predicted - red.bnb.train$price)^2)


################################################################################
## Finalize random forest and make predictions #################################
rf.preds <- predict(rf.tuned, red.bnb.test)
preds3 <- data.frame(id = bnb.test$id, 
                     price = rf.preds)

write_csv(preds3, "AirBnB price prediction using tuned random forest.csv")


