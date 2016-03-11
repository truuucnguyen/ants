%since we only care about labels for 0 and 1, shorten test_y to only look
%at the first two members
load mnist_uint8
short_test_y = zeros(10000, 2);
for i = 1:10000
    row = test_y(i, 1:10);
    short_test_y(i, 1:2) = row(1:2);
end

zerolocation = zeros(981, 1);
zeroindex = 1;
oneindex = 1;
onelocation = zeros(1136, 1);


% do the same thing for 2, 3, and 4 

%get 100 0's and 100 1's, 100 2's 100 3's 100 4's
out_data = zeros(500, 784);
out_labels = zeros(500, 2);
outindex = 1;
testindex = 1;
zerolabel = [1, 0];
onelabel = [0, 1];
%put twolabel, threelabel, and fourlabel

out_data_test = zeros(100, 784);
out_labels_test = zeros(100, 2);

for i=1:10000
    if isequal(short_test_y(i, 1:5), zerolabel)
        if zeroindex < 101
            out_data(outindex, 1:784) = test_x(i, 1:784);
            zerolocation(zeroindex) = i;
            out_labels(outindex, 1:5) = zerolabel;
            outindex = outindex + 1;
            zeroindex = zeroindex + 1;
        end
        if zeroindex >= 101 && zeroindex < 121
            out_data_test(testindex, 1:784) = test_x(i, 1:784);
            out_labels_test(testindex, 1:5) = zerolabel;
            testindex = testindex + 1;
            zeroindex = zeroindex + 1;
        end
    end
    if isequal(short_test_y(i, 1:2), onelabel)
        if oneindex < 101
            out_data(outindex, 1:784) = test_x(i, 1:784);
            onelocation(oneindex) = i;
            out_labels(outindex, 1:5) = onelabel;
            outindex = outindex + 1;
            oneindex = oneindex + 1;
        end
        if oneindex >= 101 && oneindex < 121
            out_data_test(testindex, 1:784) = test_x(i, 1:784);
            out_labels_test(testindex, 1:5) = onelabel;
            testindex = testindex + 1;
            oneindex = oneindex + 1;
        end
    end
end
    
%100 1's and 100 0's are now in out_labels and out_data
%10 1's and 10 0's are now in out_labels_test and out_data_test

%prepare output variables
ants_learn_labels = out_labels;
ants_test_labels = out_labels_test;
ants_learn_data = zeros(500, 100);
ants_test_data = zeros(100, 100);

%fill ants_learn_data
for i=1:500
    row = out_data(i,1:784);
    %unvectorize data back to 28 x 28
    unvec = reshape(row, [28 28]);
    %rotate the image
    rotated = unvec.';
    %resize image to 5 x 5
    resized = imresize(rotated, [10 10], 'bilinear');
    %zero-pad array to 10 x 10, keeping the first 5 x 5 as the data
    %padded = padarray(resized, [5 5], 0, 'post');
    %vectorize the zero-padded array to 100x1
    paddedvec = reshape(resized, [100 1]);
    %paddedvec = reshape(resized, [100 1]);
    %add this row to out_padded_data
    ants_learn_data(i, 1:100) = paddedvec;
end

%fill ants_test_data
for i=1:100
    row = out_data_test(i,1:784);
    %unvectorize data back to 28 x 28
    unvec = reshape(row, [28 28]);
    %rotate the image
    rotated = unvec.';
    if i > 10 && i < 20
        figure
        imshow(rotated);
    end
    %resize image to 10 x 10
    resized = imresize(rotated, [10 10], 'bilinear');
    %zero-pad array to 20 x 20, keeping the first 10 x 10 as the data
    %padded = padarray(resized, [5 5], 0, 'post');
    %vectorize the zero-padded array to 400x1
    paddedvec = reshape(resized, [100 1]);
    %paddedvec = reshape(resized, [100 1]);
    %add this row to out_padded_data
    ants_test_data(i, 1:100) = paddedvec;
end

%test the learn data - uncomment if you want to show all the one's that
%appear in the data

% for i=1:200
%     row = ants_learn_data(i, 1:400);
%     unvec = reshape(row, [20 20]);
%     if isequal(ants_learn_labels(i, 1:2), onelabel)
%         imshow(unvec, [0 255]);
%     end
% end
