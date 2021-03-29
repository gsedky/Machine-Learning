%% training script
% find the estimated averages and covariances under gaussian assumptions of
% the trianing data and maximum likelihood
clear; clc; close all;
% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images_train = loadMNISTImages('Data/train-images-idx3-ubyte');
labels_train = loadMNISTLabels('Data/train-labels-idx1-ubyte');
mu_hat_mat = [];
Epsilon_hat_cell= {};
% number of labels
labels_num = 10;
% PCA 
% Find the prinicipal components
coeff = pca(images_train');
% Number of dimensions kept
m = 150;
% reduce the dimensionality of the system to 20
images_Transf = coeff(:,1:m)'*images_train;
for jj=1:labels_num
images_label = images_Transf(:,labels_train==jj-1);
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
save('Estimates.mat','mu_hat_mat','Epsilon_hat_cell','coeff','m');