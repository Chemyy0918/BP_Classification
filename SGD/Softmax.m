function y = Softmax(x)
%x->List[float]:原列表
%y->List[float]:经Softmax优化过的比例列表
sum=0;
[len,~]=size(x);
y=zeros(size(x));
for i=1:len
    y(i)=exp(x(i));
    sum=sum+exp(x(i));
end
    y=y/sum;
end