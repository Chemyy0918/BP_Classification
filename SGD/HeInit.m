function Y = HeInit(mu,sigma,r,c)
%mu->float：期望
%sigma->float：标准差
%r->int：行数
%c->int：列数
%Y->Matrix:r行c列的高斯分布矩阵
Y=normrnd(mu,sigma,r,c);
end

