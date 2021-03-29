%% Train SVM on reduced dimension FMNIST
clear; clc; close all;
% Girguis Sedky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the reduced data file
load('Data_PCA.mat');

%% Train SVM using different kernels
% linear kernel
model_linear = svmtrain(labels_train, images_train', '-t 0');
% polynomial kernel
model_poly = svmtrain(labels_train, images_train', '-t 1');
% radial basis function kernel
model_rbf = svmtrain(labels_train, images_train', '-t 2');

%% Save the model
save('SVM_Model.mat','model_linear','model_poly','model_rbf');