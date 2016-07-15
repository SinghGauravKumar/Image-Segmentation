function [Mdl, YpredTrain, YpredTest, Conf_mat_Train, metricTrain,...
    Conf_mat_Test, metricTest] = NB_implementation(Xtrain, Ytrain, ...
    Xtest, Ytest)
z =clock;
num_train=size(Xtrain,1);
num_test=size(Xtest,1);
classes=unique(Ytrain);
num_classes=length(classes);
fprintf('Training Naive Bayes on %d examples ...\n',num_train);
fprintf('Training data has %d classes ...\n',num_classes);
fprintf('The classes are [%s] ...\n',sprintf('%d ', classes));

Mdl=fitcnb(Xtrain,Ytrain);
fprintf('Naive Bayes Training completed ...\n');
fprintf('Generating Predictions ...\n');
YpredTrain=predict(Mdl,Xtrain);
YpredTest=predict(Mdl,Xtest);

[Conf_mat_Train,metricTrain]=ConfusionMatrix(YpredTrain,Ytrain,classes);
fprintf('The model fits the training data with %0.2f percent accuracy ...\n',...
    metricTrain.accuracy*100);
fprintf('Recall for the training data is found to be %0.2f percent ...\n\n',...
    metricTrain.recall*100)
[Conf_mat_Test,metricTest]=ConfusionMatrix(YpredTest,Ytest,classes);
fprintf('Testing the estimated model on %d examples ...\n',num_test);
fprintf('The model fits the testing data with %0.2f accuracy ...\n',...
    metricTest.accuracy*100);
fprintf('Recall for the testing data is found to be %0.2f percent ...\n\n\n\n',...
    metricTest.recall*100);
fprintf('Time taken in Naive Bayes implementation = %0.4f seconds ...\n\n\n\n'...
    ,etime(clock, z));
end