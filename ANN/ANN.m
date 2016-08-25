function [beta, trainerr, trainout4, trainhid2, trainw42, varargout] = ANN(trainMAT, iter, alpha, varargin)
%Build a ANN with 10 output nodes, 1 hidden layer, 3 hidden nodes, and 8 input nodes
%This is the element loop version, very slow, I also implemeted vectorized
%version, see ANN_new.m

%beta           all the weights, separated into a struct(input to hidden and
%               hidden to output.
%trainerr       training error change
%trainout4      4th output node change
%trainhid2      2nd hidden node change
%trainw42       w42 change
%varargout      including testing error change, 4th output node change, 2nd hidden node change
%trainMAT       training matrix
%iter           iteration times
%alpha          learning rate
%varargin       optional testing matrix

%Random generate initial W (totally 67)
 rng(171);
 w1 = -1 + 2 * rand(1, 27);
 rng(2);
 w2 = -1 + 2 * rand(1, 40);
 
 a2 = zeros(1, 3);  %activation function of hidden layer
 a3 = zeros(1, 10); %activation funtion of output layer
 
 %initial training output variables
 trainerr = zeros(1, iter); 
 trainout4 = zeros(1, iter);
 trainhid2 = zeros(1, iter);
 trainw42 = zeros(1, iter);
 
 %initial testing output variables
 if nargin > 0
     varargout{1} = zeros(1, iter); %test error
     varargout{2} = zeros(1, iter); %4th output node change for test data
     varargout{3} = zeros(1, iter); %2nd hidden node change for test data
 end
 
 for m = 1:iter
    %Use stochastic gradient desecent and back propagation
    for i = 1:size(trainMAT, 1)
        
        %Update hidden and output node activation function results
        a2 = updhidden(i, a2, w1, trainMAT);
        a3 = updoutput(a3, a2, w2);
    
        delta = zeros(1, 10);
        %back propgation 
        for l = 1:length(w2) %update w2
            l1 = ceil(l/4); %k in wkj
            l2 = mod(l, 4);  %j in wkj
            l2(l2 == 0) = 4;
            obs = [1, a2];
            delta(l1) = (trainMAT(i, l1) - a3(l1)) * (1-a3(l1)) * a3(l1);
            grad = - delta(l1) * obs(l2);
            w2(l) = w2(l) - alpha * grad;
        end
        
        for n = 1:length(w1) %update w1
            n1 = ceil(n/9); %j in wji
            n2 = mod(n, 9);  %i in wji
            n2(n2 == 0) = 9;
            obs = [1, trainMAT(i, 11:end)];
            wkj = w2((n1 + 1) : 4 : 40);
            grad = -(delta * wkj') *(1 - a2(n1)) * a2(n1) * obs(n2);
            w1(n) = w1(n) - alpha * grad;
        end
    end
    %Calculate training error, 4th output node change, 2nd hidden node change. w42 change 
    correct = 0;
    
    for i1 = 1:size(trainMAT, 1)
        a2 = updhidden(i1, a2, w1, trainMAT); %Update a2 and a3 again for each observation.
        a3 = updoutput(a3, a2, w2);
        [~, predclass] = max(a3);
        trueclass = find(trainMAT(i1, 1:10) == 1);
        correct = correct + (trueclass == predclass); 
    end
    err = (size(trainMAT, 1) - correct) / size(trainMAT, 1);
    trainerr(m) = err;
    trainout4(m) = a3(4);
    trainhid2(m) = a2(2);
    trainw42(m) = w2(15);
    
    %Calculate test error, 4th output node change, 2nd hidden node change. w42 change 
    if nargin > 0
        testMAT = varargin{1};
        correct = 0;
        
        for i2 = 1:size(testMAT, 1)
            a2 = updhidden(i2, a2, w1, testMAT);
            a3 = updoutput(a3, a2, w2); %Updating activation function values in each layer
            [~, predclass] = max(a3);
            trueclass = find(testMAT(i2, 1:10) == 1);
            correct = correct + (trueclass == predclass);
        end
        err = (size(testMAT, 1) - correct) / size(testMAT, 1);
        varargout{1}(m) = err;
        varargout{2}(m) = a3(4);
        varargout{3}(m) = a2(2);
    end

 end
 beta = struct('weightinhid', w1, 'weighthidout', w2);
end

function a2up = updhidden(i, a2, w1, MAT)
    a2up = zeros(1, length(a2));
    for j = 1:length(a2)  %update hidden layer
        obs = [1, MAT(i, 11:end)];
        a2up(j) = actfun(obs, w1(9*j-8 : 9*j));
    end
end

function a3up = updoutput(a3, a2, w2)
    a3up = zeros(1, length(a3));
    for k = 1:length(a3) %update output layer
        obs = [1, a2];
        a3up(k) = actfun(obs, w2(4*k-3 : 4*k));
    end
end

function activation = actfun(obs, beta) %Calculating activation function
    activation = 1/(1 +exp(-(obs * beta')));
end