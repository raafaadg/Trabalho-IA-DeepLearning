clc

addpath('C:\Users\rafae\Desktop\10 ConvNET DeepLearningToolbox  v2(1)\data')
addpath('C:\Users\rafae\Desktop\10 ConvNET DeepLearningToolbox  v2(1)\CNN')
addpath('C:\Users\rafae\Desktop\10 ConvNET DeepLearningToolbox  v2(1)\util')

load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x  = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y  = double(test_y');

figure(1);
title('training samples - INPUTS')
for i = 1:20
    subplot(4,5,i);
    imshow(train_x(:,:,i))
    
    [position,aux]  = find(train_y(:,i)>0);
    label(i)        = position - 1;
    title(['Label: ' num2str(label(i))]);
end

figure(2);
title('test samples - INPUTS')
for i = 1:20
    subplot(4,5,i);
    imshow(test_x(:,:,i))
end