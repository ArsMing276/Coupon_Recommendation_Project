function [bestlambda, Mse] = CV_lambda(w0, h0, sparse_mat, lambda, alpha)
%w0                               initial w
%h0                                initial h
%sparse_mat                  the rating matrix, first column is row number, second
%                                    column is column number, third column is corresponding values 
%lambda                        optimization penalty
%alpha                           learning rate

rng(171)
MSE = zeros(1, length(lambda));
for j = 1:length(lambda)
    lbd = lambda(j);    
    mse =0;
    cvfold = crossvalind('Kfold', sparse_mat(:, 3), 10);
    for k = 1:10 %ten fold cross-validation
        w = w0; h = h0;
        testmat = sparse_mat(cvfold == k, :);
        trainmat = sparse_mat(cvfold ~= k, :);
        randnum = randi(size(trainmat, 1), 1, 10000);
        
        for i = 1:10000 %apply stochastic gradient descent to update both w and h alternatively.
            idx = randnum(i);
            row = trainmat(idx, 1); col = trainmat(idx, 2); val = trainmat(idx, 3);
            nw = sum(trainmat(:, 1) == row);
            nh = sum(trainmat(:, 2) == col); %find out number of non-zero values in the specific row and col
            wrow = w(row, :); hcol = h(:, col);
            w(row, :) = (1 - alpha * lbd / nw) .* wrow - alpha * (wrow * hcol - val) .* hcol';
            h(:, col) = (1 - alpha * lbd/ nh) .* hcol - alpha * (wrow * hcol - val) .* wrow';
        end
        
        newmat = w * h;
        ind = sub2ind(size(newmat), testmat(:, 1), testmat(:, 2));
        newval = newmat(ind); %get the predicting values for testing data
        mse = mse + (testmat(:, 3) - newval)' * (testmat(:, 3) - newval);
    end
    MSE(j) = mse/size(sparse_mat, 1);
end
[~,I] = min(MSE);
Mse = min(MSE);
bestlambda = lambda(I);
end
