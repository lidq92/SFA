function [feature1, feature2, feature3] = SFAfeature(im_lists, layer_name, options)
% SFA features

% written by Dingquan Li
% dingquanli@pku.edu.cn
% IDM, SMS, PKU
% Last update: Aug. 9, 2017

if ~exist('options','var')
    options = 'none';
end

% multi-patch representation
patchSize = [224 224]; %
stride = floor(patchSize/2); %

% Initialize the DCNN
model = [fileparts(which(mfilename)) '/models/ResNet-50-deploy.prototxt']; %
weights = [fileparts(which(mfilename)) '/models/ResNet-50-model.caffemodel']; %
caffe.set_mode_gpu(); %
net = caffe.Net(model, weights, 'test');

l = net.blobs(layer_name).shape; 
l = l(3);
feature1 = zeros(length(im_lists),2*l);
feature2 = zeros(length(im_lists),5*l);
feature3 = zeros(length(im_lists),4*l);
for k = 1:length(im_lists)
    fprintf(['Extracting the feature of the ' num2str(k) 'th image ...\n']);
    
    im = imread(im_lists{k});
    
    patches = im2patches(im, patchSize, stride);

    % feature extraction
    patch_features = DCNN(patches, net, layer_name);

    % feature aggregation
    feature_mean = mean(patch_features,2);
    feature_std = std(patch_features,0,2);

    feature_q4 = prctile(patch_features,100,2);
    feature_q3 = prctile(patch_features,75,2);
    feature_q2 = prctile(patch_features,50,2);
    feature_q1 = prctile(patch_features,25,2);
    feature_q0 = prctile(patch_features,0,2);

    feature_Moment2 = moment(patch_features,2,2);
    feature_Moment3 = moment(patch_features,3,2);
    feature_Moment4 = moment(patch_features,4,2);

    feature1(k,:) = [feature_mean;feature_std]'; % mean&std aggregation
    feature2(k,:) = [feature_q4;feature_q3;feature_q2;feature_q1;feature_q0]'; % quantile aggregation
    feature3(k,:) = [feature_mean; nthroot(feature_Moment2,2);nthroot(feature_Moment3,3);...
        nthroot(feature_Moment4,4)]'; % moment aggregation
    
    
    switch options
        case 'flipH'
            patches = im2patches(im(:,end:-1:1,:), patchSize, stride);

            % feature extraction
            patch_features = DCNN(patches, net, layer_name);

            % feature aggregation
            feature_mean = mean(patch_features,2);
            feature_std = std(patch_features,0,2);

            feature_q4 = prctile(patch_features,100,2);
            feature_q3 = prctile(patch_features,75,2);
            feature_q2 = prctile(patch_features,50,2);
            feature_q1 = prctile(patch_features,25,2);
            feature_q0 = prctile(patch_features,0,2);

            feature_Moment2 = moment(patch_features,2,2);
            feature_Moment3 = moment(patch_features,3,2);
            feature_Moment4 = moment(patch_features,4,2);

            feature1(k+length(im_lists),:) = [feature_mean;feature_std]'; % mean&std aggregation
            feature2(k+length(im_lists),:) = [feature_q4;feature_q3;feature_q2;feature_q1;feature_q0]'; % quantile aggregation
            feature3(k+length(im_lists),:) = [feature_mean; nthroot(feature_Moment2,2);nthroot(feature_Moment3,3);...
                nthroot(feature_Moment4,4)]'; % moment aggregation
        case 'clipUL'
            T = 6;
            for i = 0:T-1
                for j = 0:T-1
                    if i^2+j^2==0
                        continue
                    end
                    patches = im2patches(im(i+1:end,j+1:end,:), patchSize, stride);

                    % feature extraction
                    patch_features = DCNN(patches, net, layer_name);

                    % feature aggregation'res5c'
                    feature_mean = mean(patch_features,2);
                    feature_std = std(patch_features,0,2);

                    feature_q4 = prctile(patch_features,100,2);
                    feature_q3 = prctile(patch_features,75,2);
                    feature_q2 = prctile(patch_features,50,2);
                    feature_q1 = prctile(patch_features,25,2);
                    feature_q0 = prctile(patch_features,0,2);

                    feature_Moment2 = moment(patch_features,2,2);
                    feature_Moment3 = moment(patch_features,3,2);
                    feature_Moment4 = moment(patch_features,4,2);

                    feature1(k+(i*T+j)*length(im_lists),:) = [feature_mean;feature_std]'; % mean&std aggregation
                    feature2(k+(i*T+j)*length(im_lists),:) = [feature_q4;feature_q3;feature_q2;feature_q1;feature_q0]'; % quantile aggregation
                    feature3(k+(i*T+j)*length(im_lists),:) = [feature_mean; nthroot(feature_Moment2,2);nthroot(feature_Moment3,3);...
                        nthroot(feature_Moment4,4)]'; % moment aggregation 
                end
            end
    end
end
caffe.reset_all(); 

function patch_features = DCNN(patches, net, layer_name)
% Extract high-level semantic features from pool5 of ResNet-50

% Preprocess the patches
im_data = patches(:,:,[3 2 1],:); % RGB2BGR
im_data = permute(im_data,[2 1 3 4]); % HWC2WHC
im_data = single(im_data); % single

cropped_dim = size(patches,1);
mean_data = caffe.io.read_mean([fileparts(which(mfilename)) '/models/imagenet_mean.binaryproto']); %
topleft = floor((size(mean_data,1)-cropped_dim)/2)+1;
mean_data = mean_data(topleft:topleft+cropped_dim-1,topleft:topleft+cropped_dim-1,:);

crops_data = arrayfun(@(i)im_data(:,:,:,i)-mean_data,1:size(patches,4),'UniformOutput',false);

% Extract features
l = net.blobs(layer_name).shape; % 2048
l = l(3);
patch_features = zeros(l,length(crops_data));
for n = 1:length(crops_data)
    net.forward(crops_data(n));
%     patch_features(:,n) = std(reshape(net.blobs('res5c').get_data(),[],l,1),0,1); % pool5
    patch_features(:,n) = mean(reshape(net.blobs(layer_name).get_data(),[],l,1)); % pool5
%     patch_features(:,n) = max(reshape(net.blobs('res5c').get_data(),[],l,1)); % pool5
end
patch_features = double(patch_features);

function [patches,imcol,I] = im2patches(I, patchSize, stride)
% IM2PATCHES Extract rectangular patches from input image.
%
%   patches = IM2PATCHES(I,patchSize) I can be a MxN matrix (e.g. gray-
%   scale image) or a MxNxK array (e.g. RGB image). patchSize can be:
%   a) a scalar, in which case: patchHeight = patchWidth = patchSize.
%   b) a vector [patchHeight,patchWidth]
%   c) a Mx4 matrix with bounding box coordinates in the form
%   [xmin,ymin,xmax,ymax].
%   patches is a patchHeight x patchWidth x K x nPatches array (K >= 1).
% 
%   patches = IM2PATCHES(I,patchSize,stride) stride between consecutive
%   patches. Stride can be used to extract overlapping patchesand can be
%   either a scalar, or a 2x1 vector to define different strides across
%   axes x and y. 
% 
%   [patches,imcol,I] = IM2PATCHES(...) also returns imcol, which is a
%   nPixelsPerPatch x nPatches matrix whose columns are the elements of
%   each patch, and I, which is the input image I after padding with zeros.
% 
%   USAGE EXAMPLES:
%   patches = im2patches(I,[h,w]);   % equal to: im2col(I,[h,w],'distinct')
%   patches = im2patches(I,[h,w],1); % equal to: im2col(I,[h,w],'sliding')
% 
%   NOTE: IM2PATCHES extracts patches padding with zeros when necessary.
%   The only exception is when stride == 1, or when the stride has a value
%   that the last patches horizontally and vertically fit precicely in the
%   image, without crossing the borders.
% 
% See also: patches2im, im2col, col2im
% 
% Stavros Tsogkas, <stavros.tsogkas@ecp.fr>
% Last update: August 2015 

assert(ismatrix(I) || ndims(I) == 3, 'Input must be a 2D or 3D array');
if ismatrix(patchSize) && size(patchSize,2) == 4
    warning('This part of the function has not been tested')
    % patchSize is a Mx4 matrix containing the [xmin,ymin,xmax,ymax]
    % coordinates of bounding boxes that are fully contained in the image
    bb      = patchSize;
    bb(:,1) = max(1,bb(:,1));
    bb(:,2) = max(1,bb(:,2));
    bb(:,3) = min(size(I,2),bb(:,3));
    bb(:,4) = min(size(I,1),bb(:,4));
    patches = cell(size(bb,1),1);
    for i=1:size(bb,1)
        patches{i} = I(bb(i,2):bb(i,4),bb(i,1):bb(i,3),:);
    end
else
    assert(all(patchSize > 0), 'Patch size cannot be negative or zero')
    if isscalar(patchSize)          % Square patches
        patchHeight = patchSize;
        patchWidth  = patchSize;
    elseif numel(patchSize) == 2    % Rectangular patches
        patchHeight = patchSize(1);
        patchWidth  = patchSize(2);
    else
        error('patchSize can be either a scalar or a [patchHeight, patchWidth] vector.')
    end
    if nargin < 3                   % Identical strides for X and Y axis.
        strideX = patchWidth;       % Default is equivalent to 'distinct' 
        strideY = patchHeight;      % option for Matlab's im2col.
    elseif isscalar(stride)
        strideY = stride;
        strideX = stride;
    elseif numel(stride) == 2       % Different strides for X and Y axis.
        strideY = stride(1);
        strideX = stride(2);
    else
        error('Stride can be either a scalar or a [strideX, strideY] vector.')
    end
    [hin,win,din] = size(I); 
    hout = hin - mod(hin-patchHeight,strideY) + strideY*(strideY > 1);
    wout = win - mod(win-patchWidth, strideX) + strideX*(strideX > 1);
    nPixelsPerPatch = patchWidth*patchHeight*din;
    I(end+1:hout,end+1:wout,:) = 0; % pad with zeros if necessary
    % x-y indices for a single (possibly N-dimensional) patch 
    [x,y,z] = meshgrid(1:patchWidth,1:patchHeight,1:din);
    % pixel indices for all patches
    [xstart,ystart] = meshgrid(0:strideX:(wout-patchWidth),0:strideY:(hout-patchHeight));
    inds    = bsxfun(@plus, reshape(y,nPixelsPerPatch,[]), ystart(:)');
    inds    = inds + (bsxfun(@plus, hout*reshape(x-1,nPixelsPerPatch,[]), hout*xstart(:)')); 
    inds    = bsxfun(@plus, inds, (z(:)-1)*(hout*wout));
    imcol   = I(inds);
    patches = reshape(imcol,patchHeight,patchWidth,din,[]);
end
% =========================================================================
% Alternative versions for Matlab's im2col 'distinct' and 'sliding' modes.
% These versions are a little faster than the code used in im2patches for
% creating the imcol image, but they cannot handle an arbitrary stride
% between neighboring patches.
% =========================================================================
% % -------------------------------------------------------------------------
% function out = im2col_distinct(A,blocksize)
% % -----------------------------------------------------------------------
% nrows = blocksize(1);
% ncols = blocksize(2);
% nele = nrows*ncols;
% 
% row_ext = mod(size(A,1),nrows);
% col_ext = mod(size(A,2),ncols);
% 
% padrowlen = (row_ext~=0)*(nrows - row_ext);
% padcollen = (col_ext~=0)*(ncols - col_ext);
% 
% A1 = zeros(size(A,1)+padrowlen,size(A,2)+padcollen);
% A1(1:size(A,1),1:size(A,2)) = A;
% 
% t1 = reshape(A1,nrows,size(A1,1)/nrows,[]);
% t2 = reshape(permute(t1,[1 3 2]),size(t1,1)*size(t1,3),[]);
% t3 =  permute(reshape(t2,nele,size(t2,1)/nele,[]),[1 3 2]);
% out = reshape(t3,nele,[]);
% 
% return;
% 
% % -------------------------------------------------------------------------
% function out = im2col_sliding(A,blocksize)
% % -------------------------------------------------------------------------
% 
% nrows = blocksize(1);
% ncols = blocksize(2);
% 
% %// Get sizes for later usages
% [m,n] = size(A);
% 
% %// Start indices for each block
% start_ind = reshape(bsxfun(@plus,[1:m-nrows+1]',[0:n-ncols]*m),[],1); %//'
% 
% %// Row indices
% lin_row = permute(bsxfun(@plus,start_ind,[0:nrows-1])',[1 3 2]);  %//'
% 
% %// Get linear indices based on row and col indices and get desired output
% out = A(reshape(bsxfun(@plus,lin_row,[0:ncols-1]*m),nrows*ncols,[]));
% 
% return; 