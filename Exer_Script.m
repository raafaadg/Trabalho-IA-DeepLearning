close all
clear all
clc

addpath('.\10 ConvNET DeepLearningToolbox  v2')
addpath('.\10 ConvNET DeepLearningToolbox  v2\data')
addpath('.\10 ConvNET DeepLearningToolbox  v2\CNN')
addpath('.\10 ConvNET DeepLearningToolbox  v2\util')
addpath('.\10 ConvNET DeepLearningToolbox  v2\tests')
addpath('.\HandWritten-Digit-Recognition')
addpath('.\MLP-MATLAB')
addpath('.\MLP-MATLAB\data')
addpath('.\MLP-MATLAB\mnistHelper')
addpath('.\MLP-MATLAB\NN')

test_example_CNN_v3(3);
test_example_CNN_v3(4);
test_example_CNN_v3(5);
test_example_CNN_v3(6);
test_example_CNN_v3(7);
nntest(3);
[ Wlr, blr, Wnn1, Wnn2, bnn1, bnn2, trainErr, ...
    testCorr, testWrong, ErrPercent] = proj3()