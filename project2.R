rm(list = ls())

getwd()

x = c("DMwR", "usdm", "caret", "randomForest", "e1071",
      "DataCombine", "doSNOW", "inTrees", "rpart.plot", "rpart",'MASS','stats')
lapply(x, require, character.only = TRUE)
rm(x)

train = read.csv('train_cab.csv', header = T)
test = read.csv('test.csv', header = T)

train$fare_amount = as.numeric(as.character(train$fare_amount))
train$passenger_count = round(train$passenger_count)

#Removing fare_amount < 1
train = train[-which(train$fare_amount < 1),]

#Removing passenger_count > 6 and passenger_count < 1

train = train[-which(train$passenger_count < 1),]
train = train[-which(train$passenger_count > 6),]

#Removing pickup_latitude > 90 and and lats and longs = 0
train = train[-which(train$pickup_latitude > 90),]
train = train[-which(train$pickup_longitude == 0),]
train = train[-which(train$dropoff_longitude == 0),]

df = train

unique(train$passenger_count)
unique(test$passenger_count)
train[,'passenger_count'] = factor(train[,'passenger_count'], labels=(1:6))
test[,'passenger_count'] = factor(test[,'passenger_count'], labels=(1:6))

train = train[complete.cases(train),]


pl1 = ggplot(train,aes(x = factor(passenger_count),y = fare_amount))
pl1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)


vals = train[,"fare_amount"] %in% boxplot.stats(train[,"fare_amount"])$out
train[which(vals),"fare_amount"] = NA

sum(is.na(train$fare_amount))


train = train[complete.cases(train),]


train$pickup_date = as.Date(as.character(train$pickup_datetime))
train$pickup_weekday = as.factor(format(train$pickup_date,"%u"))# Monday = 1
train$pickup_mnth = as.factor(format(train$pickup_date,"%m"))
train$pickup_yr = as.factor(format(train$pickup_date,"%Y"))
pickup_time = strptime(train$pickup_datetime,"%Y-%m-%d %H:%M:%S")
train$pickup_hour = as.factor(format(pickup_time,"%H"))


test$pickup_date = as.Date(as.character(test$pickup_datetime))
test$pickup_weekday = as.factor(format(test$pickup_date,"%u"))# Monday = 1
test$pickup_mnth = as.factor(format(test$pickup_date,"%m"))
test$pickup_yr = as.factor(format(test$pickup_date,"%Y"))
pickup_time = strptime(test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test$pickup_hour = as.factor(format(pickup_time,"%H"))


sum(is.na(train))
train = na.omit(train)

train = subset(train,select = -c(pickup_datetime,pickup_date))
test = subset(test,select = -c(pickup_datetime,pickup_date))


haversine = function(long1,lat1,long2,lat2){
  
  rad <- pi/180
  a1 <- lat1*rad
  a2 <- long1*rad
  b1 <- lat2*rad
  b2 <- long2*rad
  dlon <- b2 - a2
  dlat <- b1 - a1
  a <- (sin(dlat/2))^2 + cos(a1)*cos(b1)*(sin(dlon/2))^2
  c <- 2*atan2(sqrt(a), sqrt(1 - a))
  R <- 6378137
  d <- R*c
  return(d)
}

train$dist_travel = haversine(train$pickup_longitude,train$pickup_latitude,train$dropoff_longitude,train$dropoff_latitude)
test$dist_travel = haversine(test$pickup_longitude,test$pickup_latitude,test$dropoff_longitude,test$dropoff_latitude)

train = subset(train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))


train = subset(train,select=-pickup_weekday)
test = subset(test, select= -pickup_weekday)


train[,'dist_travel'] = (train[,'dist_travel'] - min(train[,'dist_travel']))/(max(train[,'dist_travel'] - min(train[,'dist_travel'])))


set.seed(123)
tr.idx = createDataPartition(train$fare_amount,p=0.8,list = FALSE)
train_data = train[tr.idx,]
test_data = train[-tr.idx,]


#Linear Regression Model
lm_model = lm(fare_amount ~.,data=train_data)
summary(lm_model)
lm_predictions = predict(lm_model,test_data[,2:6])
regr.eval(test_data[,1],lm_predictions)
# mae       mse      rmse      mape 
#3.194839 16.470469  4.058383  0.422415


#Decision Tree
Dt_model = rpart(fare_amount ~ ., data = train_data, method = "anova")
summary(Dt_model)
predictions_DT = predict(Dt_model, test_data[,2:6])
regr.eval(test_data[,1],predictions_DT)
#mae       mse      rmse      mape 
#1.7471868 5.7122745 2.3900365 0.2152477



#Random Forest Model
rf_model = randomForest(fare_amount ~.,data=train_data)
summary(rf_model)
rf_predictions = predict(rf_model,test_data[,2:6])
regr.eval(test_data[,1],rf_predictions)
# mae       mse      rmse      mape 
#1.7184590 5.4094430 2.3258209 0.2184212


#Ridge Regression
x <- model.matrix(fare_amount ~., data = train_data)[,-1]
y <- train_data$fare_amount
cv <- cv.glmnet(x,y, alpha = 0)
cv$lambda.min
ridge_model <- glmnet(x,y,alpha = 0,lambda = cv$lambda.min)
x.test <- model.matrix(fare_amount ~., data = test_data)[,-1]
ridge_predictions <- predict(ridge_model,x.test)
regr.eval(test_data$fare_amount, predictions)
#mae        mse       rmse       mape 
#3.1948919 16.4705332  4.0583905  0.4224492


#Lasso Regression
cv_lasso = cv.glmnet(x,y,alpha = 1)
cv_lasso$lambda.min
lasso_model = glmnet(x,y,alpha = 1, lambda = cv_lasso$lambda.min)
lasso_predictions = predict(lasso_model, x.test)
regr.eval(test_data$fare_amount, lasso_predictions)
# mae        mse       rmse       mape 
#3.1947357 16.4699926  4.0583239  0.4224056 


#Since the random forest model has the most suitable metric values
#Random forest will be used to predict data of the test.csv file

predicted_fare_amount = data.frame(predict(rf_model, test))
names(predicted_fare_amount)[1] = 'predicted_fare_amount'
