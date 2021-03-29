%% Reduce the dimension of the FMNIST data using PCA
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

%% PCA 
% Find the prinicipal components
coeff = pca(images_train');
% Number of dimensions kept
m = 50;

%% reduce the dimensionality of the system to m
images_train = coeff(:,1:m)'*images_train;  
images_test = coeff(:,1:m)'*images_test;  

%% Save the data
save('Data_PCA.mat','images_train','labels_train','images_test','labels_test');