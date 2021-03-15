%% 该代码为基于卷积神经网络的手写体识别
% function: CNN.m
%% 清空环境变量
% 清空环境变量， 清空控制台
clear all;
clc;
% 加载数据集
load("MNISTData.mat");

%% 设置学习速率 alpha
% 生成权值矩阵
alpha = 0.01;
W1 = randn(9, 9, 20);
W3 = (2*rand(100, 2000) - 1) / 20;
W4 = (2*rand(10, 100) - 1) / 10;

%% 打乱数据及顺序，使其排列更加离散
permIndex = randperm(size(X_Train, 3));
X_Train = X_Train(:, :, permIndex);
D_Train = D_Train(:, permIndex);

%% 定义时间戳
tic 
t3 = toc
%% 训练模型
for k = 1:2 % 两轮训练
    for i=1:60000
        x = X_Train(:,:,i); 
        d = D_Train(:, i);
        V1 = Conv(W1, x);
        Y1 = ReLU(V1);
        Y2 = Pool(Y1);  % 池化
        y2 = reshape(Y2,[], 1);
        % 前向
        v3 = W3 * y2; y3 = ReLU(v3);
        v = W4 * y3; y = Softmax(v);
        % 反向
        e = d - y;
        delta = e;
     
        e3 = W4' * delta;
        delta3 = (v3 > 0).*e3;
    
        e2 = W3' * delta3;
        E2 = reshape(e2, size(Y2));
        E1 = zeros(size(Y1));
        E2_4 = E2/4;
        E1(1:2:end, 1:2:end, :) = E2_4;
        E1(1:2:end, 2:2:end, :) = E2_4;
        E1(2:2:end, 1:2:end, :) = E2_4;
        E1(2:2:end, 2:2:end, :) = E2_4;
        delta1 = (V1 > 0).*E1;
        % 更新权值矩阵
        dW4 = alpha*delta*y3';
        W4 = W4 + dW4;
    
        dW3 = alpha*delta3*y2';
        W3 = W3 + dW3;
    
        dW1 = alpha*Conv(delta1, x);  
        W1 = W1 + dW1;
    end
end
t4 = toc - t3

%% 测试模型
N = length(D_Test);
d_comp = zeros(1,N);
for k=1:N
   X = X_Test(:,:,k);
   V1 = Conv(W1, X);
   Y1 = ReLU(V1);
   Y2 = Pool(Y1);
    y2 = reshape(Y2, [], 1);
    v3 = W3 * y2; y3 = ReLU(v3);
    v = W4 * y3; y = Softmax(v);
    [~, i] = max(y);
    d_comp(k) = i;
end
% 正确率计算
[~,d_true] = max(D_Test);
acc = sum(d_true == d_comp);
fprintf("Accuracy is %f\n", acc/N);