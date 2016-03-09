% This code implements convolutional tensor decomposition
% copyright Furong Huang, furongh@uci.edu
% Cite paper arXiv:1506.03509
% This function estimates the filters based on conf.sample.

clear;clc;
addpath('fn-2d/')
L = 1;
%load(['../data/syntheticData_2d_L',num2str(L),'_estimate.mat']);
%load(['../data/FruitDatasampling',num2str(L),'_estimate.mat']);
load(['../data/syntheticData_50_2d_Lnew2_estimate.mat']);

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
save(['../data/FruitDatasampling',num2str(L),'_estimate.mat'],'conf','estimate');
