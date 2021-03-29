%% Nearest Neighborhood
clear; clc; close all;
% Girguis Sedky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
% Prototypes
load('LDA_Data.mat');

%% Loop throuogh test data
for j = 1:length(images_test)
    % Pick image
    image = images_test(:,j);
    % Find eucliden distance to all the training set
    Dist = vecnorm(images_train-image,2,1); 
    [~,I] = min(Dist);
    labels_est(j) = labels_train(I);
end
%% Test accuracy of classifying algorithm
a = find(labels_est==labels_test');
Performance = length(a)*100/length(labels_test);
save('Performance.mat','Performance')