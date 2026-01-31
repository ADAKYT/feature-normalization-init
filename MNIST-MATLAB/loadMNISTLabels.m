%% chNorm

function labels = loadMNISTLabels(filename)

fid = fopen(filename,'rb');
assert(fid~=-1,'File not found');

magic = fread(fid,1,'int32',0,'ieee-be');
assert(magic==2049,'Invalid MNIST label file');

labels = fread(fid,inf,'unsigned char');
fclose(fid);

labels = double(labels);
end
