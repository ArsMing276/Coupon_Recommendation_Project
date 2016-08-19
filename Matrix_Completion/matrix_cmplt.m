%This is the function to find the five best coupons for each user after all parameters have been determined 
%after cross-validation.

function [w,h,recmat] = matrix_cmplt(w0, h0, sparse_mat, lambda, alpha)
%w0                               initial w
%h0                                initial h
%sparse_mat                  the rating matrix, first column is row number, second
%                                    column is column number, third column is corresponding values 
%lambda                        optimization penalty
%alpha                           learning rate
%recmat                         pick out the five coupons with highest scores(indice)
    
    w = w0; h = h0;
    n = size(sparse_mat, 1);
    randnum = randi(n , 1, 1000000);
    for i = 1:1000000
        idx = randnum(i);
        row = sparse_mat(idx, 1); col = sparse_mat(idx, 2); val = sparse_mat(idx, 3);
        nw = sum(sparse_mat(:, 1) == row);
        nh = sum(sparse_mat(:, 2) == col);
        wrow = w(row, :); hcol = h(:, col);
        w(row, :) = (1 - alpha * lambda / nw) .* wrow - alpha * (wrow * hcol  - val) .* hcol';
        h(:, col) = (1 - alpha * lambda / nh) .* hcol  - alpha * (wrow * hcol  - val) .* wrow';
    end
    
    newmat = w * h;
    ind = sub2ind(size(newmat), sparse_mat(:, 1), sparse_mat(:, 2));
    newmat(ind) = sparse_mat(:, 3);
    [~, I] = sort(newmat, 2);
    I = fliplr(I);
    recmat = I(:, 1:5);
end
