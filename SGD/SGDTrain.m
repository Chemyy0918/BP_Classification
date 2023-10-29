function [CellofWeights,CellofBias] = SGDTrain(Classification,ListOfHiddenLayers,DataSet,ita,NumOfEpoch,minibatchsize)
%{
Classification->int:分类目标数;
ListOfHiddenLayer->List[int]:包括隐藏层信息的横数组,如[5,4,3]表示有三个隐藏层,分别有5、4、3个维度;
DataSet->Matrix[float|int]:训练数据集,要求同一行属于同一个数据组，且最后一列的数表示分类类别序号，例如最后一列数仅有1、2、3，表示分为三类，分别是1号类、2号类、3号类;
ita->float:学习率，一般0.01~0.1
NumOfEpoch->int：训练轮数
minibatchsize:minibatch大小
CellofWeights->Cell[Matrix[float]]：训练完的权重矩阵元胞
CellofBias->Cell[Matrix[float]]：训练完的偏置矩阵元胞
%}
[~,M]=size(DataSet);%M:输入层维度+1;N:样本总数;
M=M-1;%去掉分类指标那列就是原数据维数
[~,NumOfHiddenLayers]=size(ListOfHiddenLayers);%获得隐藏层个数
ListOfNet=[M ListOfHiddenLayers Classification];%整个网络的层信息矩阵
CellofWeights=cell(1,NumOfHiddenLayers+1);%初始化权重矩阵元胞,其元素数应=NumOfHiddenLayers+1
CellofBias=cell(1,NumOfHiddenLayers+1);%初始化偏置向量元胞,其元素数应=NumOfHiddenLayers+1
for i=1:NumOfHiddenLayers+1%初始化步骤
    CellofWeights{i}=HeInit(0,sqrt(sqrt(2/ListOfNet(i))),ListOfNet(i+1),ListOfNet(i));%按He初始化方法初始化权重矩阵元胞
    CellofBias{i}=HeInit(0,sqrt(sqrt(2/ListOfNet(i))),ListOfNet(i+1),1);%按He初始化方法初始化偏置向量元胞
end
hold on
for times=1:NumOfEpoch%主循环
    [Train_Loss,Valid_Loss,CellofWeights,CellofBias]=SGDModel(DataSet,NumOfHiddenLayers,CellofWeights,CellofBias,ita/(1+ita*exp(times-1)),Classification,minibatchsize);
    scatter(times,Train_Loss,"b*")
    scatter(times,Valid_Loss,"r*")
    pause(0.000001)
end