%% Nearest Neighborhood, with PCA, investigate the effect of different components
clear; clc; close all;
% Girguis Sedky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
% Prototypes
images_train = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\train-images-idx3-ubyte');
labels_train = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\train-labels-idx1-ubyte');
% Test data
images_test = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Data\t10k-labels-idx1-ubyte');
% PCA 
% Find the prinicipal components
coeff = pca(images_train');
% Number of dimensions kept
m = [10:20:150];
%% Loop over different values of m, number of dimensions kept
for ii=1:length(m)
% reduce the dimensionality of the system to 20
images_train_Transf = coeff(:,1:m(ii))'*images_train;  
images_test_Transf = coeff(:,1:m(ii))'*images_test;  
    %% Loop throuogh test data
for j = 1:length(images_test_Transf)
    % Pick image
    image = images_test_Transf(:,j);
    % Find eucliden distance to all the training set
    Dist = vecnorm(images_train_Transf-image,2,1);
    [~,I] = min(Dist);
    labels_est(j) = labels_train(I);
end
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