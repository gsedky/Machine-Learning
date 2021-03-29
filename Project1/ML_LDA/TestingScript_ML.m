%% Test script
% Test the ML algorthim based on gaussian assumptions of
% the training data and maximum likelihood
clear; clc; close all;
% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
load('LDA_Data.mat')
load('Estimates.mat');
% number of labels
labels_num = 10;
m=labels_num-1;
for ii=1:length(images_test)
% pick an image to classify
image = images_test(:,ii);
for jj=1:labels_num
% Find the posterior probablity estimate of each one
cov = Epsilon_hat_cell{jj};
mu =  mu_hat_mat(:,jj);
P(jj) = (1/(sqrt(det(cov)*(2*pi)^m)))*exp(-0.5*(image-mu)'*inv(cov)*(image-mu));
end
[~,I] = max(P);
labels_est(ii) = I-1;
clear P;
end

% Test accuracy of classifying algorithm
a = find(labels_est==labels_test');
Performance = length(a)*100/length(labels_test);
save('Performance.mat','Performance');