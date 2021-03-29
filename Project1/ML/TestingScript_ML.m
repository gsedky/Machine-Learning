%% Test script
% Test the ML algorthim based on gaussian assumptions of
% the training data and maximum likelihood
clear; clc; close all;
% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data/t10k-images-idx3-ubyte');
labels = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data/t10k-labels-idx1-ubyte');
load('Estimates.mat');
% number of labels
labels_num = 10;
for ii=1:length(images)
% pick an image to classify
image = images(:,ii);
% n = number of images
n = size(image,2);
% L = number of pixels in an image
L = size(image,1);
for jj=1:labels_num
% Find the posterior probablity estimate of each one
cov = Epsilon_hat_cell{jj};
cov_det = cov+(10^-10)*eye(size(cov));  % Make covariance matrix full rank so that the logdet function can find a determinant
mu =  mu_hat_mat(:,jj);                 % average vector
% Find the likelihood of this class
logP(jj) = -0.5*logdet(cov_det)-0.5*(image-mu)'*pinv(cov)*(image-mu);

end
% Choose the macimym likelihood and assign the image to this class
[~,I] = max(logP);
labels_est(ii) = I-1;
clear P;
end
labels_est = labels_est';
% Test accuracy of classifying algorithm
a = find(labels_est==labels);
Performance = length(a)*100/length(labels);
save('Perfomance.mat','Perfomance');