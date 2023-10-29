clear
clc
I=[29	32	306	61	288];
Input=1./-exp(I(1:5));
load("W.mat")
load("b.mat")
[P,O]=SGDpredict(Input',W,b);%（待分类向量，W矩阵，b向量）
[n,~]=size(O);
for i=1:n
    fprintf("属于第%d个的概率为%f\n",i,O(i));
end
fprintf("最有可能是第%d类",P);