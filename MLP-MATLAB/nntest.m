function []=nntest(setup_CNN)
clc
addpath('./NN');
addpath('./mnistHelper');
addpath('./data');

load_data;

epochs = 10;
batchsize = 250;
learning_rate = 1e-3;
iters_per_epoch = 1000;

nn = Network(batchsize);

smooth_loss = log(10);

% setup_CNN = input('Setup (1-original, 2- Trab Option 1, 3- Trab Option 2):');
switch setup_CNN
    case 1,   % original    (MSE = 0.113, epoch 1, time 137.633908 s)
        v = 256;
    case 2,   % proposed #1  (MSE = 0.108700, epoch 1, time 210.785619 s)
        v = 128;
    case 3, % trabalho #1  (MSE = 0.108200, epoch 1, time 111.994223 s) ok
        v = 512;
end
% ~ 98 % correct
nn.layers{1} = Linear(28 * 28, v, batchsize);
nn.layers{2} = ReLU(v, v, batchsize);
nn.layers{3} = Linear(v, 10, batchsize);
nn.layers{4} = Softmax(10, 10, batchsize);

tic;
for e=1:1:epochs

    fprintf('Epoch %d...\n', e);
    
    for ii=1:1:iters_per_epoch

    	nn.train(train_images, train_labels, learning_rate);
    	smooth_loss = [smooth_loss smooth_loss(end) * 0.99 + 0.01 * nn.loss/batchsize];
    
    end
    
    nn.test(test_images, test_labels);

    figure(1);
    plot(smooth_loss);
    drawnow

end
toc;
end




    