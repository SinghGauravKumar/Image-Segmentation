function newmat=norm_mat(mat)
newmat=zeros(size(mat));
minmat=min(mat);
maxmat=max(mat);
for i=1:size(mat,2)
    newmat(:,i)=(mat(:,i)-minmat(i))/(maxmat(i)-minmat(i));
end
end