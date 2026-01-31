%% PyTorch nn.Linear default initialization (MATLAB equivalent)
clc; clear;

%% Define layer sizes
in_features  = 4;
out_features = 3;
use_bias     = true;

%% Allocate parameters (same shapes as PyTorch)
W = zeros(out_features, in_features);   % weight
if use_bias
    b = zeros(out_features, 1);         % bias
else
    b = [];
end

%% === Kaiming He Uniform Initialization ===
% PyTorch uses:
% kaiming_uniform_(weight, a=sqrt(5))

a = sqrt(5);
fan_in = in_features;

% Bound derived from Kaiming uniform
bound_W = sqrt(6 / fan_in) / a;

% Initialize weights
W = (2 * bound_W) .* rand(out_features, in_features) - bound_W;

%% === Bias Initialization ===
if use_bias
    bound_b = 1 / sqrt(fan_in);
    b = (2 * bound_b) .* rand(out_features, 1) - bound_b;
end

%% === Display results ===
disp('Initialized weight matrix W:');
disp(W)

if use_bias
    disp('Initialized bias vector b:');
    disp(b)
end

%% === Simple forward pass (to see output) ===
x = rand(in_features, 1);   % sample input
y = W * x + (use_bias * b);

disp('Input x:');
disp(x)

disp('Output y = Wx + b:');
disp(y)
