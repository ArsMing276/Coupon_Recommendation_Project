%Cross Validation for Matrix completion to find the best combination of
%latent feature space for W, H; Regularized penalty and learning rate. 
function [best_latent, best_lambda, best_rate, best_mse, MSE_mat] = CV_matrix_cmp(latent, lambda, rate, sparse_mat)
%latent                potential latent feature space(an arrary)
%lambda             potential penalty (an arrary)
%rate                   potential learning rate (an array)
%sparse_mat       we redesign the spase rating matrix with first column as
%                         row number, second column as column number, third column as non-zero number
%best_latent, best_lambda, best_rate      best combination

nuser = max(sparse_mat(:, 1));
ncoupon = max(sparse_mat(:, 2));

MSE_mat = zeros(length(latent), length(rate)); %smallest mse for each pair of latent space and learnin rate
lambda_mat = zeros(length(latent), length(rate)); %best lambda for each pair of latent space and learnin rate

for i = 1:length(latent)
    for j = 1:length(rate)
        w0 = rand(nuser, latent(i)) .* 2 -1;
        h0 = rand(latent(i), ncoupon) .* 2 -1; %randomly generate initial W, H
        
        %We split the lambda selection process into an individual function because some times we are pretty 
        %sure which latent space, learning rate to use (pre-knowledge) thus we could use this sub function
        %directly without having to consider latent space and learning rate selection.
        [bestlambda, MSE] = CV_lambda(w0, h0, sparse_mat, lambda, rate(j));
        MSE_mat(i, j)  = MSE;
        lambda_mat(i, j) = bestlambda; %some values may be NAN or Inf 
    end    
end

[best_mse, ind] = min(MSE_mat(:));
[I, J] = ind2sub(size(MSE_mat),ind);
best_latant = latent(I); best_rate = rate(J);
best_lambda = lambda_mat(I, J);

end
