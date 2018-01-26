% SFA-PLSR Demo

% written by Dingquan Li
% dingquanli@pku.edu.cn
% IDM, SMS, PKU
% Last update: May 17, 2017

caffe_path = '/home/ldq/LDQ/Software/caffe/matlab/'; % point to the caffe path
addpath(genpath(caffe_path)); 

for i = 1:4
    im = imread(['examples/I04_08_' num2str(i) '.bmp']);
    quality(i) = SFA(im); %  use this function in its directory
end
