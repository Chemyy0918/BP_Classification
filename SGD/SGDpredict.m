function [PClass,Output] = SGDpredict(Input,CellOfWeights,CellOfBias)
%{
Input->List[float]:输入向量
CellOfWeights->Cell[Matrix[float]]：训练完的权重矩阵元胞
CellOfBias->Cell[Matrix[float]]：训练完的偏置矩阵元胞
PClass->int：预测数据的分类指标
Output->List[float]:预测数据在各指标下的概率输出
%}
[~,n]=size(CellOfWeights);
Output=Input;
for i=1:n
    if i~=n
        Output=ReLU(CellOfWeights{i}*Output+CellOfBias{i});
    else
        Output=Softmax(CellOfWeights{i}*Output+CellOfBias{i});
    end
end
[~,PClass]=max(Output);
end