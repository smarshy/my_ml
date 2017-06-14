% Spam classification | Support Vector Machines


%% Initialization
clear ; close all; clc

%% =============== Loading and Visualizing Data ================

fprintf('Loading and Visualizing Data ...\n')

% Load from example dataset 1, get X, y in your environment
load('data1.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Training Linear SVM ====================
%  Train a linear SVM on the dataset and plot the decision boundary learned.
%

% Load from example dataset 1, get X, y in your environment
load('data1.mat');

fprintf('\nTraining Linear SVM ...\n')

% YChange the C value below to see how the decision boundary varies (e.g., try C = 1000)
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Implementing Gaussian Kernel ===============
%  Gaussian kernel to use with SVM
%
fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :' ...
         '\n\t%f\n(this value should be about 0.324652)\n'], sim);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Visualizing Dataset 2 ================

fprintf('Loading and Visualizing Data ...\n')

% Load from example dataset 2, get X, y in your environment
load('data2.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Training SVM with RBF Kernel (Dataset 2) ==========

fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

% Load from example dataset 2, get X, y in your environment
load('data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Visualizing Dataset 3 ================

fprintf('Loading and Visualizing Data ...\n')

% % Load from example dataset 3, get X, y in your environment
load('data3.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Training SVM with RBF Kernel (Dataset 3) ==========

%  Different values of C and sigma here

% Load from example dataset 3, get X, y in your environment
load('data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

