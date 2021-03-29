%% ML with PCA
% find the estimated averages and covariances under gaussian assumptions of
% the training data and maximum likelihood, utilize PCA
clear; clc; close all;
% Girguis Sedky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
images_train = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\train-images-idx3-ubyte');
labels_train = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\train-labels-idx1-ubyte');
images_test = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\t10k-labels-idx1-ubyte');
mu_hat_mat = [];
Epsilon_hat_cell= {};
% number of labels
labels_num = 10;
% PCA 
% Find the prinicipal components
coeff = pca(images_train');
% Number of dimensions kept
m = [10:20:150];
%% Loop over different values of m, number of dimensions kept
for ii=1:length(m)
%% Training Stage
% reduce the dimensionality of the system to 20
images_Transf = coeff(:,1:m(ii))'*images_train;
%% Loop over all the pictures in one class inside the training set
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
%% Test Stage
for kk=1:length(images_test)
% pick an image to classify
image = images_test(:,kk);
% reduce the dimensionality of the system to 20
image_Transf = coeff(:,1:m(ii))'*image;
for jj=1:labels_num
% Find the posterior probablity estimate of each one
COV = Epsilon_hat_cell{jj};
mu =  mu_hat_mat(:,jj);
P(jj) = (1/(sqrt(det(COV)*(2*pi)^m(ii))))*exp(-0.5*(image_Transf-mu)'*inv(COV)*(image_Transf-mu));
end
[~,I] = max(P);
labels_est(kk) = I-1;
clear P;
end
clear mu_hat_mat Epsilon_hat_cell;
%% Test accuracy of classifying algorithm
a = find(labels_est==labels_test');
Performance(ii) = length(a)*100/length(labels_test);

end
%% save and plot
save('Performance.mat','m','Performance');
plot(m,100-Performance,'LineWidth',1.5);
xlabel('Number of components');
ylabel('% error');
grid on;
PrintPlot(gcf,'plot.png','-dpng');
PrintPlot(gcf,'plot.pdf','-dpdf');