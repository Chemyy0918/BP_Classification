function Y = GetExVector(Class,n)
%Class->int：分类数量
%n->int：类别标号
ExVec=zeros(Class,1);
ExVec(n,1)=1;
Y=ExVec;
end

