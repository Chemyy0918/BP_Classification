function y = dReLUForOne(x)
%x->float
%y->float:将一个x按dReLU规则映射为一个dy/dx
y=1-tanh(x)^2;
end