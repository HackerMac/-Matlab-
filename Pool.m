%% 该代码为基于卷积神经网络的手写体识别
% function: Pool.m
%% 清空环境变量
function OutputArg = Pool(Y1)
    OutputArg = (Y1(1:2:end,1:2:end,:) + Y1(2:2:end,1:2:end,:)...
    + Y1(1:2:end, 2:2:end,:) + Y1(2:2:end,2:2:end,:))/4;
end

