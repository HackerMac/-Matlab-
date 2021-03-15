%% 该代码为基于卷积神经网络的手写体识别
% function: Relu.m
%% 清空环境变量
function y = ReLU(x)
  y = max(0, x);
end