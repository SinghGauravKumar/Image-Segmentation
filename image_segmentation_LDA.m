clear
close all
clc
Xtrain=dlmread('segmentation.data',',',5,1);
Xtest=dlmread('segmentation.test',',',5,1);
num_cols=size(Xtrain,2)+1;
Ytrain=extract_labels(num_cols,'segmentation.data',4);
Ytest=extract_labels(num_cols,'segmentation.test',4);
Xtrain(:,3)=[];
Xtest(:,3)=[];
Xtrain=norm_mat(Xtrain);
Xtest=norm_mat(Xtest);
[w,pc,ev]=pca(Xtrain);
var=cumsum(ev/sum(ev));
w1=w(:,1:5);
Xtrain=Xtrain*w1;
Xtest=Xtest*w1;

rng(1);
P1=randperm(length(Xtrain),length(Xtrain));
P2=randperm(length(Xtest),length(Xtest));
Xtrain=Xtrain(P1,:);
Ytrain=Ytrain(P1,:);
Xtest=Xtest(P2,:);
Ytest=Ytest(P2,:);

% [Xtrain,Xtest]=swap(Xtrain,Xtest);
% [Ytrain,Ytest]=swap(Ytrain,Ytest);

[W, YpredTrain, YpredTest, Conf_mat_Train, metricTrain,...
    Conf_mat_Test, metricTest] = LDA_implementation(Xtrain, Ytrain, ...
    Xtest, Ytest);
NumNeighbors=5;
[Mdl, YpredTrain, YpredTest, Conf_mat_Train, metricKNNTrain,...
    Conf_mat_Test, metricKNNTest] = kNN_implementation(NumNeighbors,Xtrain,...
    Ytrain, Xtest, Ytest);

[Mdl, YpredTrain, YpredTest, Conf_mat_Train, metricTrain,...
    Conf_mat_Test, metricTest] = NB_implementation(Xtrain, Ytrain, ...
    Xtest, Ytest);