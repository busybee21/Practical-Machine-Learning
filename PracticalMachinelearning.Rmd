---
title: "Practical Machine Learning"
author: "Snehadrita Das"
date: "2022-08-21"
output: html_document
---

## Executive Summary  

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  In this project, our goal was to use the data rom accelerometers on the belt, forearm, arm, and dumbell of 6 participants and predict the manner in which they did the exercise. We have considered *nested models* and let them compete for greater accuracy and lesser out of sample error. We have considered machine learning algorithms such as *Decision Trees, Random Forest and Gradient Boosted Trees*. Among the models, the model with Random Forest as its algorithm predicts with the highest accuracy of 97.4% and 2.6% of out of sample error.   

## Background and Data  

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).   

* [Train Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)  
* [Test Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)  

## Getting Data  

```{r,comment=NA,cache=TRUE}
url1<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

destfile1<-"C:/Users/Hp/Desktop/train.csv"
destfile2<-"C:/Users/Hp/Desktop/test.csv"

download.file(url1,destfile1)
download.file(url2,destfile2)

traindata<-read.csv("train.csv")  ## training data
testdata<-read.csv("test.csv")    ## test data

dim(traindata);dim(testdata)
```

### Library Dependency  

```{r,comment=NA,message=FALSE}
library(corrplot)
library(caret)
library(rpart.plot)
library(randomForest)
library(gbm)
```

## Data Processing  

* We shall explore the training dataset for missing values and omit them. We shall omit the variables with loads of mussing values. To be noted that all the processing and working do be done in the training data set. The test data is only for prediction.  
* We shall also omit the variables with very smaller (near zero) variance as they contribute little to the model and prediction procedure.  

```{r,comment=NA,message=FALSE,cache=TRUE}
na<-rep(0,160)
for(i in 1:160){
     na[i]=sum(is.na(traindata[,i]))  ## looking for NA values and storing them in the vector na
}
idx<-which(na==19216)                ## indices for the variables with mostly NA values
traindata<- traindata[,-c(1:7,idx)]  ## processing 
near.zero<-nearZeroVar(traindata)     ## indices for the variables with mostly lesser variances
final.traindata<-traindata[,-near.zero]  ## Processed training set

dim(final.traindata)
```

## Train set and Validation set  

```{r,comment=NA,message=FALSE}
set.seed(17)
intrain<-createDataPartition(final.traindata$classe,p=0.7,list=FALSE)
train.set<-final.traindata[intrain,]   ## train set
validation<-final.traindata[-intrain,]  ## validation set

dim(train.set);dim(validation)
```

## Exploratory Data Analysis  

* We shall see the association between the variables in the train set  

```{r,message=FALSE,comment=NA}
corrplot(cor(train.set[,-53]),method = "color",type = "lower",
                   tl.cex = 0.6, tl.col = rgb(0, 0, 0),main="Correlation Plot")

```

## Cross Validation  
Since Cross Validation is a very powerful tool and  a statistical method used to estimate the performance (or accuracy) of machine learning models and is used to protect against overfitting in a predictive model, particularly in a case where the amount of data may be limited, we shall be considering cross valiadtion as our train control.  

```{r,comment=NA}
## cross validation
control<-trainControl(method = "cv",number = 3)
```

## Algorithms and Models  

### ***Decision Tree***  

### Model  

```{r,comment=NA,message=FALSE}
model.dt<-train(classe~.,data=train.set,method="rpart",trControl=control)
rpart.plot(model.dt$finalModel)
```

### Prediction and Accuracy  

```{r,comment=NA,message=FALSE}
dt.pred<-predict(model.dt,newdata = validation)  ## prediction
dt.mat<-confusionMatrix(dt.pred,factor(validation$classe))  ## accuracy
dt.mat
dt.mat$overall
```

### ***Random Forest***  

### Model  

```{r,comment=NA,message=FALSE,cache=TRUE}
## random forest
model.rf=train(classe~.,data=train.set,method="rf",trControl=control,prox=TRUE,ntree=4)
model.rf$finalModel
```

### Prediction and Accuracy  

```{r,comment=NA,message=FALSE,cache=TRUE}
rf.pred<-predict(model.rf,newdata = validation)
rf.mat<-confusionMatrix(rf.pred,factor(validation$classe))
rf.mat
rf.mat$overall
```

### ***Gradient Boosted Trees***  

### Model  

```{r,comment=NA,message=FALSE,cache=TRUE}
model.gbm<-train(classe~.,data=train.set,method="gbm",trControl=control,verbose=FALSE)
print(model.gbm$finalModel)
```

###  Prediction and Accuracy   

```{r,comment=NA,message=FALSE,cache=TRUE}
gbm.pred<-predict(model.gbm,newdata = validation)
gbm.mat<-confusionMatrix(gbm.pred,factor(validation$classe))
gbm.mat
gbm.mat$overall
```


## Comparisons  

```{r,echo=FALSE,comment=NA,message=FALSE}
Algorithm=c("Decision Tree","Random Forest","GBM")
Accuracy=c(dt.mat$overall[1],rf.mat$overall[1],gbm.mat$overall[1])
Out.of.sample.Error=c(1-dt.mat$overall[1],1-rf.mat$overall[1],1-gbm.mat$overall[1])
viz=data.frame(Algorithm,Accuracy,Out.of.sample.Error)
viz
```

* ***We see that the model with Random Forest algorithm predicts with 97.4% accuracy and hence we shall use this model to perform prediction o our test data***   

## Prediction on Test data  

```{r,comment=NA}
predict(model.rf,testdata)
```


## Appendix  

* [Github](https://github.com/busybee21/Practical-Machine-Learning)

The Rmd file and the html file of this project can be found in the above Github repository.   

## Model visualization  

```{r,message=FALSE,comment=NA}
par(mfrow=c(1,2))
plot(model.rf,main="Random Forest Algorithm Model")
plot(model.gbm,main="GBM Algorithm Model")
```

### Code Chunk for the Comparison section  

```{r,echo=TRUE,comment=NA,message=FALSE,results='hide'}
Algorithm=c("Decision Tree","Random Forest","GBM")
Accuracy=c(dt.mat$overall[1],rf.mat$overall[1],gbm.mat$overall[1])
Out.of.sample.Error=c(1-dt.mat$overall[1],1-rf.mat$overall[1],1-gbm.mat$overall[1])
viz=data.frame(Algorithm,Accuracy,Out.of.sample.Error)
viz
```















