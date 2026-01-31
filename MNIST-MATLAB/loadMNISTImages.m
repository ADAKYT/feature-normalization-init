function images = loadMNISTImages(filename)

fid = fopen(filename,'rb');
assert(fid~=-1,'File not found');

magic = fread(fid,1,'int32',0,'ieee-be');
assert(magic==2051,'Invalid MNIST image file');

numImages = fread(fid,1,'int32',0,'ieee-be');
numRows   = fread(fid,1,'int32',0,'ieee-be');
numCols   = fread(fid,1,'int32',0,'ieee-be');

images = fread(fid,inf,'unsigned char');
fclose(fid);

images = reshape(images,numRows*numCols,numImages)';
images = double(images)/255;
end
