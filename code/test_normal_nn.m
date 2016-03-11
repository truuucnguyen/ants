function test_normal_nn(nn)

% While loop checks if there is a file named 'test.jpg' in the
% ../data/images folder for testing 
while (true)
    if (any(size(dir('../data/images/test.jpg'),1)))
        % Read the image
        test = imread('../data/images/test.jpg');
        
        % Prepares the image for testing
        test_g = rgb2gray(test);
        [x,y] = size(test_g);
        test_sz = min(x,y);
        test_cropped = imresize(test_sz, [test_sz test_sz]);
        test_scale = 10/test_sz;
        test_resized = imresize(test_cropped, test_scale);
        test_f = reshape(test_resized, [1 100]);
        
        %Test
        label = nnpredict(nn, test_f);
        
        %After making prediction, if it is 0 shape then raise flag in .txt file
        fid = fopen( '../data/output/output.txt', 'wt' );
        c = clock;
        c = fix(c);
        if (c(4) > 11)
            ampm = 'PM';
        else
            ampm = 'AM';
        end
        if (mod(c(4),12) == 0)
            c(4) = 12;
            ampm = 'AM';
        else 
            c(4) = mod(c(4),12);
        end
        
        if label == 0
            fprintf(fid, 'Flag: %d, Date: %d-%02d-%02d, Time: %02d:%02d:%02d %s', 1, c(1), c(2), c(3), c(4), c(5), c(6),ampm);
        else
            fprintf(fid, 'Flag: %d, Date: %d-%02d-%02d, Time: %02d:%02d:%02d %s', 0, c(1), c(2), c(3), c(4), c(5), c(6),ampm);
        end
        fclose(fid);
    end
end

end