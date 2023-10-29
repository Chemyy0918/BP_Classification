function Y=ReLU(X)
%X->List[float]
%Y->List[float]:将X按逐元素ReLU映射
Y=arrayfun(@ReLUForOne,X);
end