function B = FullPerm(A)
%A,B->Matrix[Any]:B是A的按行随机重排
B=A(randperm(size(A,1)),:);
end

