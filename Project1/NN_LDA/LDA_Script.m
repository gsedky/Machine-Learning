%% Reduce the data diemsnionality using LDA
clear; clc; close all;
% Girguis Sedky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
images_train = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\train-images-idx3-ubyte');
labels_train = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\train-labels-idx1-ubyte');
images_test = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\t10k-labels-idx1-ubyte');
% number of labels
A = LDA(images_train',labels_train);
A = A(1:end-1,1:end-1);
images_train = A*images_train;
images_test = A*images_test;
save('LDA_Data.mat','images_train','images_test','labels_train','labels_test');