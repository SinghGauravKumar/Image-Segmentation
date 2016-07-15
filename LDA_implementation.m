function [W, YpredTrain, YpredTest, Conf_mat_Train, metricTrain,...
    Conf_mat_Test, metricTest] = LDA_implementation(Xtrain, Ytrain, ...
    Xtest, Ytest)
z =clock;
% Xtrain = randi(10,25,2);
% Ytrain = ones(25,1);
% Ytrain(Xtrain(:,1)<=3 & Xtrain(:,2)>=5)=2;
% Ytrain(Xtrain(:,1)<=7 & Xtrain(:,2)<=3)=3;
num_train=size(Xtrain,1);
num_test=size(Xtest,1);
classes=unique(Ytrain);
num_classes=length(classes);
fprintf('Training LDA on %d examples ...\n',num_train);
fprintf('Training data has %d classes ...\n',num_classes);
fprintf('The classes are [%s] ...\n',sprintf('%d ', classes));

% W is num_classes X num_features+1
W = LDA(Xtrain,Ytrain);
fprintf('LDA Training completed ...\n');
fprintf('Generating Predictions ...\n');

Ltrain = [ones(num_train,1) Xtrain] * W';
Ltest = [ones(num_test,1) Xtest] * W';
Ptrain = exp(Ltrain) ./ repmat(sum(exp(Ltrain),2),[1 num_classes]);
Ptest = exp(Ltest) ./ repmat(sum(exp(Ltest),2),[1 num_classes]);
[~,jTrain]=max(Ptrain,[],2);
[~,jTest]=max(Ptest,[],2);
YpredTrain=classes(jTrain);
YpredTest=classes(jTest);
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
fprintf('Time taken in LDA implementation = %0.4f seconds ...\n\n\n\n'...
    ,etime(clock, z));
end