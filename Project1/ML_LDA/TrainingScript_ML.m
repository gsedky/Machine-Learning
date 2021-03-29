%% training script
% find the estimated averages and covariances under gaussian assumptions of
% the trianing data and maximum likelihood, with LDA
clear; clc; close all;
load('LDA_Data.mat')
mu_hat_mat = [];
Epsilon_hat_cell= {};
% number of labels
labels_num = 10;
for jj=1:labels_num
images_label = images_train(:,labels_train==jj-1);
% n = number of images
n = size(images_label,2);
% L = number of pixels in an image
L = size(images_label,1);
% Find the maximum likelihoods estimates for the mean and covariance
mu_hat = mean(images_label,2);
Epsilon_hat = cov(images_label');
mu_hat_mat(:,jj) = mu_hat;
Epsilon_hat_cell{jj} = Epsilon_hat;

end
save('Estimates.mat','mu_hat_mat','Epsilon_hat_cell');