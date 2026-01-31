%% =========================================================
% MLP with Batch Normalization on MNIST (Pure MATLAB)
% FINAL - NO ERROR VERSION
%% =========================================================

clc; clear; close all;
rng(123);

%% -----------------------
% SETTINGS
%% -----------------------
batch_size   = 128;     % کمتر برای جلوگیری از out of memory
num_epochs   = 10;
lr           = 0.1;

num_features = 28*28;
num_hidden_1 = 75;
num_hidden_2 = 45;
num_classes  = 10;

%% -----------------------
% LOAD MNIST (LOCAL FILES)
%% -----------------------
% فایل‌ها باید کنار همین کد باشند
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_images  = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels  = loadMNISTLabels('t10k-labels.idx1-ubyte');

% حتماً لیبل‌ها ستونی باشند
train_labels = train_labels(:);
test_labels  = test_labels(:);

%% -----------------------
% VALIDATION SPLIT
%% -----------------------
N = size(train_images,1);
val_n = floor(0.1*N);

X_val = train_images(1:val_n,:);
Y_val = train_labels(1:val_n);

X_train = train_images(val_n+1:end,:);
Y_train = train_labels(val_n+1:end);

clear train_images train_labels   % آزادسازی حافظه

%% -----------------------
% FUNCTIONS
%% -----------------------
relu = @(x) max(0,x);

softmax = @(x) exp(x) ./ sum(exp(x),2);

batchnorm = @(x) (x - mean(x,1)) ./ sqrt(var(x,0,1) + 1e-5);

%% -----------------------
% INITIALIZE WEIGHTS
%% -----------------------
W1 = 0.01*randn(num_features,num_hidden_1);
W2 = 0.01*randn(num_hidden_1,num_hidden_2);
W3 = 0.01*randn(num_hidden_2,num_classes);

b1 = zeros(1,num_hidden_1);
b2 = zeros(1,num_hidden_2);
b3 = zeros(1,num_classes);

%% -----------------------
% TRAINING
%% -----------------------
num_batches = ceil(size(X_train,1)/batch_size);

train_acc = zeros(num_epochs,1);
val_acc   = zeros(num_epochs,1);
loss_hist = zeros(num_epochs,1);

for epoch = 1:num_epochs

    idx = randperm(size(X_train,1));
    X_train = X_train(idx,:);
    Y_train = Y_train(idx);

    epoch_loss = 0;

    for b = 1:num_batches

        s = (b-1)*batch_size + 1;
        e = min(b*batch_size, size(X_train,1));

        X = X_train(s:e,:);
        Y = Y_train(s:e);

        % -------- Forward --------
        z1 = X*W1 + b1;
        a1 = batchnorm(z1);
        h1 = relu(a1);

        z2 = h1*W2 + b2;
        a2 = batchnorm(z2);
        h2 = relu(a2);

        logits = h2*W3 + b3;
        probs  = softmax(logits);

        % -------- One-hot --------
        Y_onehot = zeros(size(probs));
        for i = 1:length(Y)
            Y_onehot(i, Y(i)+1) = 1;
        end

        % -------- Loss --------
        loss = -mean(sum(Y_onehot .* log(probs + 1e-8),2));
        epoch_loss = epoch_loss + loss;

        % -------- Backprop --------
        dL = (probs - Y_onehot) / size(X,1);

        dW3 = h2' * dL;   db3 = sum(dL,1);
        dh2 = dL * W3';
        da2 = dh2 .* (a2 > 0);

        dW2 = h1' * da2;  db2 = sum(da2,1);
        dh1 = da2 * W2';
        da1 = dh1 .* (a1 > 0);

        dW1 = X' * da1;   db1 = sum(da1,1);

        % -------- Update --------
        W3 = W3 - lr*dW3; b3 = b3 - lr*db3;
        W2 = W2 - lr*dW2; b2 = b2 - lr*db2;
        W1 = W1 - lr*dW1; b1 = b1 - lr*db1;
    end

    loss_hist(epoch) = epoch_loss / num_batches;

    train_acc(epoch) = accuracy(X_train, Y_train, ...
        W1, W2, W3, b1, b2, b3, relu);

    val_acc(epoch) = accuracy(X_val, Y_val, ...
        W1, W2, W3, b1, b2, b3, relu);

    fprintf('Epoch %d/%d | Train: %.2f%% | Val: %.2f%% | Loss: %.4f\n',...
        epoch, num_epochs, train_acc(epoch), val_acc(epoch), loss_hist(epoch));
end

%% -----------------------
% PLOTS
%% -----------------------
figure;
plot(train_acc,'LineWidth',2); hold on;
plot(val_acc,'LineWidth',2);
xlabel('Epoch'); ylabel('Accuracy (%)');
legend('Train','Validation');
title('MLP + BatchNorm Accuracy');
grid on;

figure;
plot(loss_hist,'LineWidth',2);
xlabel('Epoch'); ylabel('Loss');
title('Training Loss');
grid on;
