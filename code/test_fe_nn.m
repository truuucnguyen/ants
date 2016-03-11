function test_fe_nn

% Image folder must contain 3 images to create conf.H for testing
% While loop checks if there is a file named 'test.jpg' in the
% ../data/images folder for testing 
while (true)
    if (any(size(dir('../data/images/test1.jpg'),1)) && any(size(dir('../data/images/test2.jpg'),1)) && any(size(dir('../data/images/test3.jpg'),1)))
        
        data = zeros(3,100);
        % Resize and zero pad the image
        for i = 1:3
            test_g = rgb2gray(imread(strcat('../data/images/test', num2str(i),'.jpg')));
            [x,y] = size(test_g);
            test_sz = min(x,y);
            test_cropped = imresize(sample_im, [test_sz test_sz]);
            test_scale = 5/test_sz;
            test_resized = imresize(test_cropped, test_scale);
            test_padded = padarray(test_resized, [5 5], 0, 'post');
            test_v = reshape(test_padded, [1 100]);
            data(i) = test_v;
        end
        
        % num of filters = 1
        conf = struct('a', 5, 'n', 10, 'N', 3, 'L', 1, 'lambda', 0, 'f', [], 'H',[], 'sample', ants_learn_data.');
        m = matfile('../data/syntheticTestData.mat', 'Writable', true);
        save('../data/syntheticTestData.mat', 'conf');
        
        % main_learn_2d.m
        % This code implements convolutional tensor decomposition
        % copyright Furong Huang, furongh@uci.edu
        % Cite paper arXiv:1506.03509 
        % This function estimates the filters based on conf.sample. 

        %clear;clc;
        L = 1;
        load(['../data/syntheticTest.mat']);
        conf.maxIter = 100;
        conf.minIter = 1;
        conf.tol = 1e-4;
        conf.IniTrue = 0;
        addpath('fn-2d/');
        Tensor = Construct_Tensor_from_Data(conf.sample, conf.N);
        estimate = ALS_2d(conf, Tensor)

        save('../data/syntheticTestData_estimate.mat','conf','estimate');
        
        % This code implements convolutional tensor decomposition
        % copyright Furong Huang, furongh@uci.edu
        % Cite paper arXiv:1506.03509
        % This function estimates the filters based on conf.sample.

        clear;clc;
        addpath('fn-2d/')

        load('../data/syntheticTestData_estimate.mat');

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
        save('../data/syntheticTestData_estimate.mat','conf','estimate');

        load('../data/syntheticTestData_estimate.mat');
        
        % Test the first image given in conf.H
        for i = 1:3
            label = nnpredict(nn, estimate.H(i,:));
        
            %After making prediction, if it is 0 shape then raise flag in .txt file
            fid = fopen( '../data/output/output.txt', 'wt' );
            c = clock;
            c = fix(c);
            if (c(4) > 11)
                ampm = 'PM';
            else
                ampm = 'AM';
            end
            if (mod(c(4) == 0))
                c(4) = 12;
                ampm = 'AM';
            else 
                c(4) = mod(c(4),12);
            end
        
            if label == 0
                fprintf(fid, '"test%d.jpg" Flag: %d, Date: %d-%02d-%02d, Time: %02d:%02d:%02d %s', i, 1, c(1), c(2), c(3), c(4), c(5), c(6),ampm);
            else
                fprintf(fid, '%d', 0);
            end
            fclose(fid);
        end
    end
end

end