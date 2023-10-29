function Y = dReLU(X)
%X->List[float]
%Y->List[float]:将X按逐元素dReLU映射
Y=arrayfun(@dReLUForOne,X);
end

