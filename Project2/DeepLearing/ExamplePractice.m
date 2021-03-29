%% Train Neural network
clear; clc; close all;
% Girguis Sedky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
% Prototypes
images_train = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Project2\Data\train-images-idx3-ubyte');
labels_train = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Project2\Data\train-labels-idx1-ubyte');
% Test data
images_test = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Project2\Data\t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Project2\Data\t10k-labels-idx1-ubyte');

%% turn vector to image array
x1 = reshape(images_train(:,1), [28 28]);
figure;
imshow(x1);
%% Create a bank of 10 linear filters
w = randn(5,5,1,10) ;
% w = [0  1 0 ;
%      1 -4 1 ;
%      0  1 0 ] ;

%% Apply the convolution operator
x2 = vl_nnconv(x1, w, []) ;

%% Apply ReLu
x3 = vl_nnrelu(x2) ;

%% Apply max pooling
x4 = vl_nnpool(x3, [2 2],'stride',2) ;

%% Do a back propagation on the convolution operator
p = randn(size(x2)) ; % projection tensor (arbitrary)
[dimage,dw,db] = vl_nnconv(x1,w,[],p) ; % backward mode (get projected derivatives)
