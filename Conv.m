%% 该代码为基于卷积神经网络的手写体识别
% function: Conv.m
%% 清空环境变量
function OutputArg = Conv(W1,x)
    for k=1:20
        OutputArg(:,:, k) = filter2(W1(:,:,k), x, 'valid');  % 滤波
    end
end

