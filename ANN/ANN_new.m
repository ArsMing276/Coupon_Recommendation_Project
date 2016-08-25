 function [beta, trerr, trout4, trhid2, trw42, varargout] = ANN_new(trainMAT, iter, alpha, varargin)
%Build a ANN with 10 output nodes, 1 hidden layer, 3 hidden nodes, and 8 input nodes
%Vectorized version, much faster. We will use this version in the following
%problems.

%beta        all the weights, separated into a struct(input to hidden and
%            hidden to output.
%trerr       training error change
%trout4      4th output node change
%trhid2      2nd hidden node change
%trw42       w42 change
%varargout   including testing error change, 4th output node change, 2nd hidden node change
%trainMAT    training matrix
%iter        iteration times
%alpha       learning rate
%varargin    optional testing matrix

%Random generate initial W within [-1, 1] (totally 67)
 rng(171);
 w1 = -1 + 2 * rand(9, 3);
 rng(2);
 w2 = -1 + 2 * rand(4, 10);
 
 %Initial training and test outputs
 trerr = zeros(1, iter);
 trout4 = zeros(1, iter);
 trhid2 = zeros(1, iter);
 trw42 = zeros(1, iter);
 
 if nargin == 4
     varargout{1} = zeros(1, iter);
     varargout{2} = zeros(1, iter);
     varargout{3} = zeros(1, iter);
 end

 for m = 1:iter
    %Use stochastic gradient desecent and back propagation
    for i = 1:size(trainMAT, 1)
        
        %update all a2 and a3 of each observation in one calculation
        a2 = 1 ./ (1 +exp(-([1, trainMAT(i, 11:end)] * w1)));
        a3 = 1 ./ (1 +exp(-([1, a2] * w2)));
        
        %back propgation
        %update w2
        delta1 = (trainMAT(i, 1:10) - a3) .*  (1 - a3) .* a3;
        grad1 = - [1, a2]' * delta1; 
        w2 = w2 - alpha * grad1;
       
            
        %update w1
        summation = delta1 * w2';
        delta2 = summation(2:4) .* (1 - a2) .* a2;
        grad2 = - [1, trainMAT(i, 11:end)]' * delta2;
        w1 = w1 - alpha *grad2;
        
    end
     
   %Calculate training error, 4th output node change, 2nd hidden node change. w42 change 
    trmat = [ones(size(trainMAT, 1), 1), trainMAT(:, 11:end)];
    a2mat = 1 ./ (1 +exp(-(trmat * w1)));
    a2matone = [ones(size(a2mat, 1), 1), a2mat];
    a3mat = 1 ./ (1 +exp(-(a2matone * w2))); %Calculating a2 and a3 values for all observations.
    [~, predclass] = max(a3mat, [], 2);
    [~, trueclass] = max(trainMAT(:, 1:10), [], 2);
    err = sum(predclass ~= trueclass) / length(predclass);
    trerr(m) = err;
    trout4(m) = a3(4);
    trhid2(m) = a2(2);
    trw42(m) = w2(3, 4);
    
   %Calculate testing error, 4th output node change, 2nd hidden node change. w42 change 
    if nargin == 4
        testMAT = varargin{1};  
        tsmat = [ones(size(testMAT, 1), 1), testMAT(:, 11:end)];
        a2mat = 1 ./ (1 +exp(-(tsmat * w1)));
        a2matone = [ones(size(a2mat, 1), 1), a2mat];
        a3mat = 1 ./ (1 +exp(-(a2matone * w2))); %Updating activation function values in each layer
        [~, predclass] = max(a3mat, [], 2);
        [~, trueclass] = max(testMAT(:, 1:10), [], 2);
        err = sum(predclass ~= trueclass) / length(predclass);
        varargout{1}(m) = err;
        varargout{2}(m) = a3mat(end, 4);
        varargout{3}(m) = a2mat(end, 2);
    end

 end
 %Output beta
 beta = struct('coefinhid', w1, 'coefhidout', w2);
end

