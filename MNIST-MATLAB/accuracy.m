function acc = accuracy(X, Y, W1, W2, W3, b1, b2, b3, relu)

    % -------- Forward --------
    Z1 = X*W1 + b1;
    A1 = relu(Z1);

    Z2 = A1*W2 + b2;
    A2 = relu(Z2);

    scores = A2*W3 + b3;

    % -------- Prediction --------
    [~, pred] = max(scores, [], 2);   % Nx1

    % -------- Fix labels --------
    if size(Y,2) > 1
        [~, Ytrue] = max(Y, [], 2);   % one-hot → Nx1
    else
        Ytrue = Y(:);                 % 1xN → Nx1
    end

    % -------- Accuracy --------
    acc = sum(pred == Ytrue) / length(Ytrue) * 100;
end
