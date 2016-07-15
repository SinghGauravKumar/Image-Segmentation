function target=extract_labels(num_cols,file,offset)
format=[];
for I=1:num_cols
if any(I==1)
 format=[format '%s'];
else
 format=[format '%*s'];
end
end
fid=fopen(file,'rt');
labels=textscan(fid,format,'delimiter',',','headerlines', offset);
fclose(fid);
labelsLIST=cellfun(@unique,labels,'UniformOutput',false);
labels=labels{1,1};
labelsLIST=labelsLIST{1,1};
target=[];
for i=1:length(labels)
    for j=1:length(labelsLIST)
        if strcmp(labels{i,1},labelsLIST{j,1})==1
            target(i,1)=j;
        end
    end
end
end