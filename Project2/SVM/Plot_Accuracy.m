% Plot the results of the SVM with different Kernels
clear; close all; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load data
load('Performance.mat');

%% create plot
X = categorical({'Linear','Polynomial','RBF'});
X = reordercats(X,{'Linear','Polynomial','RBF'});
bar(X,Performance);
ylabel('Percentage Accuracy');
ylim([80 90])
%% save
PrintPlot(gcf,'G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Project2\Documentation\Figures\SVM_Accuracy.pdf','-dpdf');

