function [Train_Loss,Valid_Loss,W,b] = SGDModel(Data,NumOfHiddenLayers,W,b,ita,Class,minibatchsize)
%{
Data->Matrix[float]：数据集
NumOfHiddenLayers->int：隐藏层数量
W->Cell[Matrix[float]]：权重矩阵元胞
B->Cell[Matrix[float]]：偏置矩阵元胞
ita->float:学习率
minibatchsize->int：mini batch的大小
Train_Loss->float：训练损失值，此处使用交叉熵误差
Valid_Loss->float：验证损失值，此处使用交叉熵误差
%}
[~,M]=size(Data);
dW=cell(size(W));
db=cell(size(b));
Train_Loss=0;
Valid_Loss=0;
NewData=FullPerm(Data);
for index=1:minibatchsize%一次训练循环（epoch）
    x=NewData(index,1:M-1)';%取出训练范围内的数据点
    ddW=cell(size(W));
    ddb=cell(size(b));
    z=cell(1,NumOfHiddenLayers+1);
    h=cell(1,NumOfHiddenLayers+1);
    delta=cell(1,NumOfHiddenLayers+1);
    z{1}=W{1}*x+b{1};%网络走进隐藏层
    h{1}=ReLU(z{1});
    for t=2:NumOfHiddenLayers+1%网络在隐藏层中以及走进输出层
        z{t}=W{t}*h{t-1}+b{t};
        if t~=NumOfHiddenLayers+1
            h{t}=ReLU(z{t});
        else
            h{t}=Softmax(z{t});
        end
    end
    Train_Loss=Train_Loss-1/Class*sum(GetExVector(Class,NewData(index,M)).*log(h{NumOfHiddenLayers+1}));
    delta{NumOfHiddenLayers+1}=h{NumOfHiddenLayers+1}-GetExVector(Class,NewData(index,M));
    for t=NumOfHiddenLayers+1:-1:2
        delta{t-1}=dReLU(z{t-1}).*((W{t})'*delta{t});
        ddW{t}=delta{t}*(h{t-1})';
        ddb{t}=delta{t};
    end
    ddW{1}=delta{1}*x';
    ddb{1}=delta{1};
    for t=1:NumOfHiddenLayers+1
        if index==1
        dW{t}=ddW{t};
        db{t}=ddb{t};
        else
        dW{t}=dW{t}+ddW{t};
        db{t}=db{t}+ddb{t};
        end
    end
end
for t=1:NumOfHiddenLayers+1
    W{t}=W{t}-ita*dW{t}/minibatchsize;
    b{t}=b{t}-ita*db{t}/minibatchsize;
end
NewData=FullPerm(Data);
for index=1:minibatchsize%一次验证循环
    x=NewData(index,1:M-1)';%取出训练范围内的数据点
    z=cell(1,NumOfHiddenLayers+1);
    h=cell(1,NumOfHiddenLayers+1);
    z{1}=W{1}*x+b{1};%网络走进隐藏层
    h{1}=ReLU(z{1});
    for t=2:NumOfHiddenLayers+1%网络在隐藏层中以及走进输出层
        z{t}=W{t}*h{t-1}+b{t};
        if t~=NumOfHiddenLayers+1
            h{t}=ReLU(z{t});
        else
            h{t}=Softmax(z{t});
        end
    end
    Valid_Loss=Valid_Loss-1/Class*sum(GetExVector(Class,NewData(index,M)).*log(h{NumOfHiddenLayers+1}));
end
Train_Loss=Train_Loss/minibatchsize;
Valid_Loss=Valid_Loss/minibatchsize;
end