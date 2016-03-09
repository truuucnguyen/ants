function train_fe_nn()
% Here we will train the resized and zeropadded images for the neural network
% Images are already preloaded into a .mat file - ants_learn_data.mat
load ../data/ants_learn_data.mat
load ../data/ants_learn_labels.mat

% Data is in ants_learn_data: 200 images of 1x400 vectorized image
% Labels are in ants_learn_labels: 200 vectors of [0,1] or [1,0]

[ants_learn_data, mu, sigma] = zscore(ants_learn_data);

rand('state',0);

% Run through furong's algorithm
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

conf = struct('a', 200, 'n', 400, 'N', 200, 'L', 1, 'lambda', 0, 'f', [], 'H',[], 'sample', ants_learn_data);

% Creates .mat file for test synthetic data and saves input_struct to it
m = matfile('../data/syntheticTest.mat', 'Writable', true);
save('../data/syntheticTest.mat', 'conf');

% main_learn_2d.m
% This code implements convolutional tensor decomposition
% copyright Furong Huang, furongh@uci.edu
% Cite paper arXiv:1506.03509 
% This function estimates the filters based on conf.sample. 

%clear;clc;
load('../data/syntheticTest.mat');
conf.maxIter = 100;
conf.minIter = 1;
conf.tol = 1e-4;
conf.IniTrue = 0;
addpath('fn-2d/');
Tensor = Construct_Tensor_from_Data(conf.sample, conf.N);
ALS(conf, Tensor)

save('../data/syntheticTest_estimate.mat','conf','estimate');

% This code implements convolutional tensor decomposition
% copyright Furong Huang, furongh@uci.edu
% Cite paper arXiv:1506.03509
% This function estimates the filters based on conf.sample.

clear;clc;
addpath('fn-2d/')

load('../data/syntheticTest_estimate.mat');

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
save('../data/syntheticTest_estimate.mat','conf','estimate');

load('../data/syntheticTest_estimate.mat');

% 2 is the number of output nodes
% 400 is number of input  nodes
% 20 is number of hidden layer nodes
nn = nnsetup([400 20 2]);
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 1;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, conf.H, ants_learn_labels, opts);

test_fe_nn(nn);
end