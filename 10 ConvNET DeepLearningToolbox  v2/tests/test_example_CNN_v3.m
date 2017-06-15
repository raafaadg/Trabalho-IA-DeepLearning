
function []=test_example_CNN_v2(setup_CNN)
% close all
% clear all
% clc
% % function test_example_CNN
% 
% % It is need to change the directory
% addpath('C:\Users\rafae\Desktop\10 ConvNET DeepLearningToolbox  v2(1)\data')
% addpath('C:\Users\rafae\Desktop\10 ConvNET DeepLearningToolbox  v2(1)\CNN')
% addpath('C:\Users\rafae\Desktop\10 ConvNET DeepLearningToolbox  v2(1)\util')


fprintf ('Convolutional neural network\n\n');

load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x  = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y  = double(test_y');

fprintf('Case study: MNIST database of handwritten digit\n');
fprintf('Dataset given by http://yann.lecun.com/exdb/mnist/');
fprintf('\nTraining with %d samples of size %dx%d\n', ...
        size(train_x,3),size(train_x,1),size(train_x,2));
fprintf('Test     with %d samples of size %dx%d\n\n', ...
        size(test_x,3),size(test_x,1),size(test_x,2));
    
% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
% will run 1 epoch in about 200 second and get around 11% error. 
% With 100 epochs you'll get around 1.2% error

rand('state',0)

% setup_CNN = input('Setup (1-original, 2- proposed, 3- Trab Option1, 4- Trab Option2, 5- Trab Option3, 6- Trab Option4):');
disp(['A opção ' num2str(setup_CNN) 'foi a selecionada']);
opts.numepochs = 1;
if setup_CNN==7
    disp(['A opção ' num2str(setup_CNN) 'foi a selecionada para rodar com 10 épocas devido seu desempenho']);
    opts.numepochs = 10;
end
fprintf('\n');

t = clock; 
switch setup_CNN
    case 1,   % original    (MSE = 0.113, epoch 1, time 137.633908 s)
        v = [6 5 12 5];
    case 2,   % proposed #1  (MSE = 0.108700, epoch 1, time 210.785619 s)
        v = [8 5 15 5];
    case 3, % trabalho #1  (MSE = 0.108200, epoch 1, time 111.994223 s) ok
        v = [9 5 17 5];
    case 4, % trabalho #1  (MSE = 0.104800, epoch 1, time 158.381356 s) ok
        v = [18 5 10 5];
    case 5, % trabalho #1  (MSE = 0.110400, epoch 1, time 119.331272 s) ok
        v = [10 5 16 7];
    case 6, % trabalho #1  (MSE = 0.110000, epoch 1, time 134.199001 s) ok
        v = [10 5 20 5];
end

cnn.layers = {
   struct('type', 'i')                                         % input layer
   struct('type', 'c', 'outputmaps', v(1), 'kernelsize', v(2)) % convolution layer
   struct('type', 's', 'scale', 2)                             % sub sampling layer
   struct('type', 'c', 'outputmaps', v(3), 'kernelsize', v(4)) % convolution layer
   struct('type', 's', 'scale', 2)                             % subsampling layer
};
  
opts.alpha     = 1;
opts.batchsize = 50;

% opts.numepochs = 1;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad, ter, tax_er] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL,'k');
ylabel('mean squared error (MSE)')

% fprintf('Mean Squared Error (MSE) = %f\n\n',er)
total_time = etime(clock, t);

file_name = strcat('Res_SetupCNN_',num2str(v(1)), ...
                '_',num2str(v(2)), ...
                '_',num2str(v(3)), ... 
                '_',num2str(v(4)), ...
                '_Epochs_', num2str(opts.numepochs),'_MSE_', ...
                num2str(er),'_Date_',datestr(clock,30),'_mat');
file_name  = strrep(file_name,'.','p');            
file_name  = strrep(file_name,'_mat','.mat')            

save (file_name, 'cnn','er','total_time')   
end
