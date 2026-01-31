%% =========================================
%  MLP on MNIST - STABLE MATLAB VERSION
%% =========================================

clc;
clear;
close all;

%% -------------------------------
% SETTINGS
%% -------------------------------
RANDOM_SEED   = 123;
NUM_EPOCHS   = 50;
BATCH_SIZE   = 256;
NUM_HIDDEN_1 = 75;
NUM_HIDDEN_2 = 45;
LEARNING_RATE = 0.1;

rng(RANDOM_SEED,'twister');

%% -------------------------------
% LOAD MNIST DATA
%% -------------------------------
[XTrain, YTrain] = digitTrain4DArrayData;
[XTest,  YTest ] = digitTest4DArrayData;

YTrain = categorical(YTrain);
YTest  = categorical(YTest);

%% -------------------------------
% TRAIN / VALID SPLIT (10%)
%% -------------------------------
numTrain = size(XTrain,4);
idx = randperm(numTrain);

numValid = floor(0.1 * numTrain);
validIdx = idx(1:numValid);
trainIdx = idx(numValid+1:end);

XValid = XTrain(:,:,:,validIdx);
YValid = YTrain(validIdx);

XTrain = XTrain(:,:,:,trainIdx);
YTrain = YTrain(trainIdx);

%% -------------------------------
% NETWORK ARCHITECTURE (MLP)
%% -------------------------------
layers = [
    imageInputLayer([28 28 1],'Normalization','none')

    fullyConnectedLayer(NUM_HIDDEN_1)
    reluLayer

    fullyConnectedLayer(NUM_HIDDEN_2)
    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

%% -------------------------------
% TRAINING OPTIONS (SGD)
%% -------------------------------
options = trainingOptions('sgdm', ...
    'InitialLearnRate', LEARNING_RATE, ...
    'MaxEpochs', NUM_EPOCHS, ...
    'MiniBatchSize', BATCH_SIZE, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValid,YValid}, ...
    'Verbose', false, ...
    'Plots','none');

%% -------------------------------
% TRAIN MODEL
%% -------------------------------
[net, info] = trainNetwork(XTrain, YTrain, layers, options);

%% -------------------------------
% TRAINING LOSS
%% -------------------------------
figure;
plot(info.TrainingLoss,'LineWidth',1.5);
xlabel('Iteration');
ylabel('Loss');
title('Training Loss');
grid on;

%% -------------------------------
% ACCURACY
%% -------------------------------
figure;
plot(info.TrainingAccuracy,'LineWidth',1.5); hold on;
plot(info.ValidationAccuracy,'LineWidth',1.5);
legend('Train','Validation');
xlabel('Iteration');
ylabel('Accuracy (%)');
ylim([80 100]);
title('Accuracy');
grid on;

%% -------------------------------
% TEST ACCURACY
%% -------------------------------
YPred = classify(net, XTest);
testAccuracy = mean(YPred == YTest) * 100;
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);
