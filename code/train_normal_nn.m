function train_normal_nn()
% Here we will train the normal images for the neural network
% Images are already preloaded into a .mat file - ants_learn_data_nopad.mat
load ../data/ants_learn_data_nopad.mat
load ../data/ants_learn_labels.mat

% Data is in ants_learn_data: 200 images of 1x400 vectorized image
% Labels are in ants_learn_labels: 200 vectors of [0,1] or [1,0]

[ants_learn_data, mu, sigma] = zscore(ants_learn_data);

rand('state',0);

% 2 is the number of output nodes
% 400 is number of input  nodes
% 20 is number of hidden layer nodes
nn = nnsetup([100 20 5]);
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 1;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, ants_learn_data, ants_learn_labels, opts);

test_normal_nn(nn);
end