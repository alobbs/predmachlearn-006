# Clean up
rm(list = ls(all = TRUE))

#Setting the working directory
#setwd('/media/nicolo/Data/Coursera/PracticalMachineLearning/Data/PracticalMachineLearning')
setwd("D:\\Coursera\\PracticalMachineLearning\\Data\\PracticalMachineLearning\\data")

# Loading the caret library
library(caret)

# Taken from the assignment to write the files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

# Reading training and testing sets
trainingRaw <- read.csv(file="pml-training.csv", header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings=c('NA','','#DIV/0!'))
testingRaw <- read.csv(file="pml-testing.csv", header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings=c('NA','','#DIV/0!'))

trainingRaw$classe <- as.factor(trainingRaw$classe)

#Removing NAs
NAindex <- apply(trainingRaw,2,function(x) {sum(is.na(x))}) 
trainingRaw <- trainingRaw[,which(NAindex == 0)]
NAindex <- apply(testingRaw,2,function(x) {sum(is.na(x))}) 
testingRaw <- testingRaw[,which(NAindex == 0)]

#Preprocess
v <- which(lapply(trainingRaw, class) %in% "numeric")

preObj <-preProcess(trainingRaw[,v],method=c('knnImpute', 'center', 'scale'))
trainLess1 <- predict(preObj, trainingRaw[,v])
trainLess1$classe <- trainingRaw$classe

testLess1 <-predict(preObj,testingRaw[,v])


# remove near zero values, if any
nzv <- nearZeroVar(trainLess1,saveMetrics=TRUE)
trainLess1 <- trainLess1[,nzv$nzv==FALSE]

nzv <- nearZeroVar(testLess1,saveMetrics=TRUE)
testLess1 <- testLess1[,nzv$nzv==FALSE]


# Create cross validation set
set.seed(12031987)
inTrain = createDataPartition(trainLess1$classe, p = 3/4, list=FALSE)
training = trainLess1[inTrain,]
crossValidation = trainLess1[-inTrain,]


# Train model with random forest
startTime <- Sys.time();

modFit <- train(classe ~., method="rf", data=training, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )

endTime <- Sys.time()
endTime - startTime


# Training set accuracy
trainingPred <- predict(modFit, training)
confusionMatrix(trainingPred, training$classe)

# Cross validation set accuracy
cvPred <- predict(modFit, crossValidation)
confusionMatrix(cvPred, crossValidation$classe)

#Predictions on the real testing set
testingPred <- predict(modFit, testLess1)
testingPred
