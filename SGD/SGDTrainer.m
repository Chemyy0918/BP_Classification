clear
clc
load("DataSet.mat")
DataSet(1:5)=1./(1-exp(DataSet(1:5)));%归一化预处理
[W,b]=SGDTrain(3,[5,4,3],DataSet,0.01,10,5000);%（目标分类数，隐藏层列表，数据集，学习率，迭代轮数，样本容量）
save("W.mat","W")
save("b.mat","b")