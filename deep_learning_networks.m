clc
clear all
close all

%% Deep learning network (Regression)

inputSize = [7 1 1];
numFeatures = 7;
miniBatchSize = 64;
maxEpochs = 300;
layers = [ ...
    imageInputLayer(inputSize,'Name','input','Normalization','none')

    convolution2dLayer([3 1],32,'Name','conv1','Padding','same')
    batchNormalizationLayer('Name','bn1') 
    reluLayer('Name','relu1')    
    
    convolution2dLayer([2 1],64,'Name','conv2','Padding','same')
    batchNormalizationLayer('Name','bn2') 
    reluLayer('Name','relu2')      

    fullyConnectedLayer(1, 'Name','fc')
    regressionLayer('Name','regression')];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.001, ...
    'L2Regularization',0.0001, ...
    'Shuffle','every-epoch', ...
    'Verbose',false,...
    'Plots','training-progress');

%% Deep learning network (Classification)

inputSize = [7 1 1];
numFeatures = 7;
numClasses = 3;
miniBatchSize = 64;
maxEpochs = 15;
layers = [ ...
    imageInputLayer(inputSize,'Name','input','Normalization','none')

    convolution2dLayer([3 1],32,'Name','conv1','Padding','same')
    batchNormalizationLayer('Name','bn1') 
    reluLayer('Name','relu1')    
    
    convolution2dLayer([2 1],64,'Name','conv2','Padding','same')
    batchNormalizationLayer('Name','bn2') 
    reluLayer('Name','relu2')      

    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.001, ...
    'L2Regularization',0.0001, ...
    'Shuffle','every-epoch', ...
    'Verbose',false,...
    'Plots','training-progress');

