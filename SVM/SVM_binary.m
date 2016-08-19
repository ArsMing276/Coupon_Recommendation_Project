%My own version of binary SVM with SMO. 
function [predclass, score, varargout] = SVM_binary(x, y, base, C, kernel, varargin)
%base           which class in y is labeled 1
%C                slack variable weight
%kernel         kernel function to be used (only linear, polynomial and RBF are defined)
%varargin      optional testdata
%predclass    prediction class for testing data if given, othereise, 
%                   predict on training data.(1 -- base, -1 -- the other class)
%predscore    SVM prediction score, positove means class1, negative means class 2
%varargout     if kernel is linear, coefficients and intercept are given, otherwise, only predictions are given.    

trainx = x; trainy = y;
if nargin == 6
    testx = varargin{1};
else
    testx = x;
end

%Three types of kernel calculation 
if strcmpi(kernel, 'linear')
    K = trainx * trainx';
    Kpred = trainx * testx';
elseif strcmpi(kernel, 'polynomial')
    K = (1 + trainx * trainx') .^ 2;  
    Kpred = (1 + trainx * testx') .^ 2;  
elseif strcmpi(kernel, 'RBF')
    X2 = repmat(reshape(diag(trainx * trainx'), 1, size(trainx, 1)), size(trainx, 1), 1);
    K = exp((X2 + X2' - 2 .* (trainx * trainx')) .* (-1));
    X_1 = repmat(reshape(diag(trainx * trainx'), size(trainx, 1), 1), 1, size(testx, 1));
    X_2 = repmat(reshape(diag(testx * testx'), 1, size(testx, 1)), size(trainx, 1), 1);
    Kpred = exp((X_1 + X_2 - 2 .* (trainx * testx')) .* (-1));
 else
    disp('Please give the right kernel, only in linear, polynomial and RBF')
end
    
 %Transform cell values into double
    yval = cellfun(@(x) strcmp(x, base), trainy, 'UniformOutput', false);
    yval = double(cell2mat(yval));
    yval(yval == 0) = -1;
    Q = diag(yval) * K * diag(yval);
    
    %initial alpha with random numbers within [0,C].
    alpha = rand(1, length(yval)) .* C;%initialize alpha
   
    %iterate 100000 times would be really fast, but for strict convergence,
    %1000000 iters are needed.
    for i = 1: 100000 %SMO update for alpha
        ind = datasample(1:length(yval), 2, 'Replace', false);
        alpha1 = alpha(ind(1));
        alpha2 = alpha(ind(2));
        sigma = alpha1 * yval(ind(1)) + alpha2 * yval(ind(2)) - alpha * yval;
        Q1 = Q(:, ind(2)) - Q(:, ind(1)) .* yval(ind(1)) .* yval(ind(2)); 
        alpha2 = ((1 - yval(ind(1)) * yval(ind(2))) - sigma * yval(ind(1)) * Q1(ind(1))...
                     - alpha * Q1 + alpha1 * Q1(ind(1)) + alpha2 * Q1(ind(2))) ./ ...
                     (Q1(ind(2)) - Q1(ind(1)) * yval(ind(1)) * yval(ind(2)));%under no constrain
    
    %find upper bound and lower bound for alpha2
       boundval1 = sigma ./  yval(ind(2));
       boundval2 = (sigma - yval(ind(1)) * C) ./ yval(ind(2));
       up = max(boundval1, boundval2);
       down = min(boundval1, boundval2);
       upper = min(up, C);
       lower = max(down, 0);
    
       alpha(ind(2)) = max(lower, min(upper, alpha2));
       alpha(ind(1)) = (sigma - alpha(ind(2)) * yval(ind(2))) * yval(ind(1));
    end
   
    %If kernel is linear, beta and intercept are given
   if strcmpi(kernel, 'linear')
       varargout{1} = (sum(bsxfun(@times, alpha' .* yval, trainx)))';
       positive = trainx(yval == 1, :); negative = trainx(yval == -1, :);
       varargout{2} = - (min(positive * varargout{1}) + max(negative * varargout{1})) / 2;
   end
   
   %Get score and class for testdata.
    score = (sum(bsxfun(@times, alpha' .* yval, Kpred)))';
    predclass = sign(score);

end
