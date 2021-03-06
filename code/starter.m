% starter.m
% Will take in a number of sample images and vectorize it
% Initially trying to scale down to around 20x20 pixels

load '../data/ants_learn_data_pad'
load '../data/ants_test_data_pad'
load '../data/ants_learn_labels'
load '../data/ants_test_labels'

% Beginning to create struct for input to Furong's algorithm
%
% a: effective filter size (int) - usually n * .5 because sparse - given
% n: filter size (int) - given = sample size
% N: number of samples (int) - given
% L: number of filters to be computed (int) - given
% lambda: weight for each component (int) - not given
% f: filter (matrix) output - initially empty
% H: activation/weight (matrix) - number of images x dimension of sample - weight
% for each image and filter where filter * w = image - empty
% sample: data of samples in vectorized form (matrix) - given
% use main_learn_2d -> main_decode_2d
% main_decode_2d will give conf.H which will be a matrix of weights to plug
% into neural network
% main_learn_2d will give conf.f and conf.lambda 
% look up mnist digit database, supervised learning 
% neural network can take in raw image, not just weights, to train
% neural network will give it true y and weights for training data
% test data will give y_hat which will show which object it matches

conf = struct('a', 5, 'n', 10, 'N', 500, 'L', 1, 'lambda', 0, 'f', [], 'H',[], 'sample', ants_learn_data.');

% Creates .mat file for test synthetic data and saves input_struct to it
m = matfile('../data/syntheticDataFE.mat', 'Writable', true);
save('../data/syntheticDataFE.mat', 'conf');

conf = struct('a', 5, 'n', 10, 'N', 100, 'L', 1, 'lambda', 0, 'f', [], 'H',[], 'sample', ants_test_data.');
m = matfile('../data/syntheticTestFE.mat', 'Writable', true);
save('../data/syntheticTestFE.mat', 'conf');

% main_learn_2d.m
% This code implements convolutional tensor decomposition
% copyright Furong Huang, furongh@uci.edu
% Cite paper arXiv:1506.03509 
% This function estimates the filters based on conf.sample. 

%clear;clc;
load('../data/syntheticDataFE.mat');
conf.maxIter = 100;
conf.minIter = 1;
conf.tol = 1e-4;
conf.IniTrue = 0;
addpath('fn-2d/');
Tensor = Construct_Tensor_from_Data(conf.sample, conf.N);
estimate = ALS_2d(conf, Tensor);
save('../data/syntheticDataFE_estimate.mat','conf','estimate');
clear;clc;

load('../data/syntheticTestFE.mat');
conf.maxIter = 100;
conf.minIter = 1;
conf.tol = 1e-4;
conf.IniTrue = 0;
addpath('fn-2d/');
Tensor = Construct_Tensor_from_Data(conf.sample, conf.N);
estimate = ALS_2d(conf, Tensor);
save('../data/syntheticTestFE_estimate.mat','conf','estimate');

% This code implements convolutional tensor decomposition
% copyright Furong Huang, furongh@uci.edu
% Cite paper arXiv:1506.03509
% This function estimates the filters based on conf.sample.

clear;clc;
addpath('fn-2d/')

load '../data/syntheticDataFE_estimate.mat'
estimate.H = zeros(size(conf.sample,2),conf.n*conf.n,conf.L);
for id_sample = 1 : size(conf.sample,2)
    fprintf('id_sample:%d\n',id_sample);
    filters = estimate.f;
    inv_concated_circulant_filters = cir_inv_2d(filters);
    thisH = inv_concated_circulant_filters*conf.sample(:,id_sample);
    for i = 1:conf.L
    estimate.H(id_sample,:,i)  = thisH((i-1)*conf.n*conf.n+1:i*conf.n*conf.n)'; 
    end
end
save('../data/mnistData_estimate.mat','conf','estimate');

load '../data/syntheticTestFE_estimate.mat'
estimate.H = zeros(size(conf.sample,2),conf.n*conf.n,conf.L);
for id_sample = 1 : size(conf.sample,2)
    fprintf('id_sample:%d\n',id_sample);
    filters = estimate.f;
    inv_concated_circulant_filters = cir_inv_2d(filters);
    thisH = inv_concated_circulant_filters*conf.sample(:,id_sample);
    for i = 1:conf.L
    estimate.H(id_sample,:,i)  = thisH((i-1)*conf.n*conf.n+1:i*conf.n*conf.n)'; 
    end
end
save('../data/mnistTest_estimate.mat','conf','estimate');
clear;clc;
load '../data/ants_learn_labels'
load '../data/mnistData_estimate.mat'
load '../data/small_ants_learn_data_nopad'
load '../data/small_ants_test_data_nopad'
%train_x = double(estimate.H);
train_x = double(small_ants_learn_data_nopad);
train_y = double(ants_learn_labels);
load '../data/ants_test_labels'
load '../data/mnistTest_estimate.mat'
%test_x = double(estimate.H);
test_x = double(small_ants_test_data_nopad);
test_y = double(ants_test_labels);

% normalize
[train_x, mu, sigma] = zscore(train_x);
%test_x = normalize(test_x, mu, sigma);


rand('state',0);
% 10 is the number of output nodes
% 100 is number of input  nodes
% 20 is number of hidden layer nodes
nn = nnsetup([100 20 2]);
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 1;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);