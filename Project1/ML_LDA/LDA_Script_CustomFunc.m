%% LDA (Page 299 of machine learning a basysean perspective)
% Reduce the dimensionality of the state vectors using LDA
clear; clc; close all;
% Girguis Sedky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
images_train = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\train-images-idx3-ubyte');
labels_train = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\train-labels-idx1-ubyte');
images_test = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\t10k-labels-idx1-ubyte');
% number of labels
labels_num = 10;
% n = number of images
n = size(images_train,2);
% L = number of pixels in an image
L = size(images_train,1);
%% LDA analysis
% intialize pertinent parameters
Cov_w = zeros(L);
Cov_b = zeros(L);
mu_0 = zeros(L,1);
mu_hat = [];
% loop to find the within catter coavriance and total average
for jj=1:labels_num
% Find mean and covariance
images_label = images_train(:,labels_train==jj-1);
mu_hat = [mu_hat,mean(images_label,2)];
Epsilon_hat = cov(images_label');
% within class scatter
Cov_w = Cov_w + Epsilon_hat;
end
% within class scatter
Cov_w = Cov_w/labels_num;
% total average
mu_0 = mean(mu_hat,2);
% loop once more to find the between class scatter
for jj=1:labels_num
Cov_b = Cov_b + (mu_hat(:,jj) - mu_0)*(mu_hat(:,jj) - mu_0)';
end
Cov_b = Cov_b/labels_num;
% find the transfromation matrix A
[lambda,A] = eig(Cov_b,Cov_w);
[lambda_diag, SortOrder]=sort(diag(lambda),'descend');
A=A(:,SortOrder);
A = A(:,1:labels_num-1);
images_train=A'*images_train;
images_test=A'*images_test;
save('LDA_Data.mat','images_train','images_test','labels_train','labels_test');