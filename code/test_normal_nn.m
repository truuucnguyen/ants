function test_normal_nn(nn)

% While loop checks if there is a file named 'test.jpg' in the
% ../data/images folder for testing 
while (true)
    if (any(size(dir('../data/images/test.jpg'),1)))
        % Test the image
        label = nnpredict(nn, test_x);
        
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
            fprintf(fid, 'Flag: %d, Date: %d-%02d-%02d, Time: %02d:%02d:%02d %s', 1, c(1), c(2), c(3), c(4), c(5), c(6),ampm);
        else
            fprintf(fid, '%d', 0);
        end
        fclose(fid);
    end
end

end