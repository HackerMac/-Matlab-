%% 该代码为基于卷积神经网络的手写体识别
% function: Softmax.m
%% 清空环境变量
function y = Softmax(x)
  ex = exp(x);
  y  = ex / sum(ex);
end