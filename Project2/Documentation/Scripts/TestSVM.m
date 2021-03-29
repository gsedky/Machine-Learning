%% Test SVM trained model on reduced dimension FMNIST test data
clear; clc; close all;
% Girguis Sedky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load the reduced data file
load('Data_PCA.mat');
load('SVM_Model.mat');

%% Run SVM on test data
[predicted_label, accuracy_linear, dec_values] = svmpredict(labels_test, images_test', model_linear);
[predicted_label, accuracy_poly, dec_values] = svmpredict(labels_test, images_test', model_poly);
[predicted_label, accuracy_rbf, dec_values] = svmpredict(labels_test, images_test', model_rbf);

%% Save the accuracy of the SVM with different kernels
Performance = [accuracy_linear(1),accuracy_poly(1),accuracy_rbf(1)];
save('Performance.mat','Performance')

