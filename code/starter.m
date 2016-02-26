% starter.m
% Will take in a number of sample images and vectorize it
% Initially trying to scale down to around 20x20 pixels
% Two different types of input: samples and filter

% Loading sample (training) image (eventually use a for loop to do this with all
% sample images
N = 11;
L = 1;
samples = zeros(N,400);
for i = 1:N
    path = strcat('../data/images/sample',num2str(i),'.jpg');
    sample_im = rgb2gray(imread(path));
    [x,y] = size(sample_im);
    sample_sz = min(x,y);
    sample_cropped = imresize(sample_im, [sample_sz sample_sz]);
    sample_scale = 20/sample_sz;
    sample_resized = imresize(sample_cropped, sample_scale);
    sample_v = reshape(sample_resized, [1 400]);
    samples(i,:) = sample_v;
end

% Loading test image
%filter_im = imread('../data/images/triangle.jpg');
filter_im = imread('../data/images/sample4.jpg');
% Changing image to grayscale
filter_gray = rgb2gray(filter_im);
% Crops the edges to create a square image
% Calculating scale for image and scales down the given image
[f_x,f_y] = size(filter_gray);
sz = min(f_x,f_y);
f_cropped = imresize(filter_gray, [sz sz]);
scale = 20/sz;
f_resized = imresize(f_cropped, scale);

% Converts image matrix into a vector (matrix with one column)
f_vectorized = reshape(f_resized, [1 400]);

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

conf = struct('a', length(f_vectorized)/2, 'n', length(f_vectorized), 'N', N, 'L', 1, 'lambda', 0, 'f', [], 'H',[], 'sample', samples);

% Creates .mat file for test synthetic data and saves input_struct to it
m = matfile(['../data/syntheticTest',num2str(L),'.mat'], 'Writable', true);
save(['../data/syntheticTest',num2str(L),'.mat'], 'conf');

% main_learn_2d.m
% This code implements convolutional tensor decomposition
% copyright Furong Huang, furongh@uci.edu
% Cite paper arXiv:1506.03509 
% This function estimates the filters based on conf.sample. 

%clear;clc;
L = 1;
load(['../data/syntheticTest',num2str(L),'.mat']);
conf.maxIter = 100;
conf.minIter = 1;
conf.tol = 1e-4;
conf.IniTrue = 0;
addpath('fn-2d/');
%Tensor = Construct_Tensor_from_Data(conf.sample, conf.N);
%ALS(conf, Tensor)

%save(['../data/syntheticTest',num2str(L),'_estimate.mat'],'conf','estimate');

train_x = double(samples);
train_y = double([[0,1];[0,1];[0,1];[1,0];[1,0];[1,0];[1,0];[1,0];[0,1];[0,1];[1,0]]);
test_x = double(f_vectorized);

% normalize
[train_x, mu, sigma] = zscore(train_x);
%test_x = normalize(test_x, mu, sigma);


rand('state',0);
% 10 is the number of output nodes
% 784 is number of input  nodes
% 100 is number of hidden layer nodes
nn = nnsetup([400 20 2]);
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 1;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);

%After making prediction, can check if shape matches to 'nothing' and
%return a flag or 0 or 1 instead of label
label = nnpredict(nn, test_x);