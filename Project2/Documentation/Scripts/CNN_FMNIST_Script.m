% train and test FMNIST data via deep learning
% Girguis Sedky
function CNN_FMNIST_Script(varargin)
setup;
%%
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------
%%
% Load training data set
% Prototypes
images_train = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Project2\Data\train-images-idx3-ubyte');
labels_train = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Project2\Data\train-labels-idx1-ubyte');
labels_train=labels_train+1;
% reshape data sp that you get a picture array
images_train = reshape(images_train, [28 28 60000]);
%%
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = initialize_FMNIST_CNN() ;
vl_simplenn_display(net)

%%
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
% Set the options for the stochastic grdaient descent back propagato
trainOpts.batchSize = 100 ;    % The number of picture that goes into one run of the descent
trainOpts.numEpochs = 15 ;     % Nummber of times the gradient procedure runs through all the data
trainOpts.continue = false ;    % If it were stopped, the descent continues where its left off
trainOpts.gpus = [] ;          % No GPU to be used
trainOpts.learningRate = 0.001 ; % Learning rate 
trainOpts.expDir = 'NNData' ;    % directory where the options are saved
trainOpts = vl_argparse(trainOpts, varargin);

% Subtract out the mean of the image
im_mean = mean(images_train(:)) ;
images_train = images_train - im_mean ;

% Create the imdb structure that is apparently needed to run this
imdb.images.data = images_train;
imdb.images.label = labels_train';
imdb.images.id = [1:1:length(imdb.images.label)];
imdb.images.set = 1*ones(1,length(imdb.images.label));
% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch, trainOpts);


% Save the result for later use
net.layers(end) = [] ;
net.imageMean = im_mean ;
save('NNData/CNN.mat', '-struct', 'net') ;

%%
% -------------------------------------------------------------------------
% Part 4.5: apply the model
% -------------------------------------------------------------------------
clear;
% Load the CNN learned before
net = load('NNData/CNN.mat') ;

% Load test data set
% Prototypes
images_test = loadMNISTImages('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Project2\Data\t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('G:\My Drive\Classes&Books\StatisticalPatternRecognition\Project\Project2\Data\t10k-labels-idx1-ubyte');
labels_test=labels_test+1;
% reshape data sp that you get a picture array
images_test = reshape(images_test, [28 28 10000]);

% subtract mean and ,multiply by that weird factor
images_test = 256 * (images_test- net.imageMean) ;

% Image to be tested
for ii=1:length(images_test)

% Apply the CNN to the test image
res = vl_simplenn(net, images_test(:,:,ii)) ;

% 
scores = squeeze(gather(res(end).x)) ;
[~, best] = max(scores);
labels_est(ii) = best;
end

labels_est = labels_est';
% Test accuracy of classifying algorithm
a = find(labels_est==labels_test);
Performance = length(a)*100/length(labels_test)
save('Perfomance.mat','Performance');
% --------------------------------------------------------------------

function [im, labels] = getBatch(imdb, batch)
%% Custo, function to create the batches that go into the descent procedure
% --------------------------------------------------------------------
im = imdb.images.data(:,:,batch) ;
im = 256 * reshape(im, 28, 28, 1, []) ;
labels = imdb.images.label(1,batch) ;

