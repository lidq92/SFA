%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     Intra Database Experiments     %%%%%
%%%%%     SFA (Li et al. TMM 2018)       %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% written by Dingquan Li
% dingquanli@pku.edu.cn
% IDM, SMS, PKU
% Last update: Sept. 15, 2018

clear;clc

caffe_path = '/home/ldq/caffe/matlab/'; % point to the caffe path
addpath(genpath(caffe_path)); 

database = 'MLIVE2';
% im_dir, im_ids, im_lists, im_names, index, ref_ids, subjective_scores, subjective_scoresSTD
load(['data/' database 'info']); % point to data info file, im_dir should be re-assigned.
% for i = 1:length(im_names) % Update im_lists when im_dir changes
%     im_lists{i} = [im_dir im_names{i}];
% end
% clear i
if ~exist('results','dir')
    mkdir('results');
end
save_path = ['./results/' database '-reproduce']; % point to save path

%% Feature Extraction
layer_names = {'res3d', 'res4f', 'res5c'};
options = 'none';
for k = 1:length(layer_names)
    layer_name = layer_names{k};
    eval(['[feature1.' layer_name ', feature2.' layer_name ', feature3.' layer_name ...
        '] = SFAfeature(im_lists, layer_name, options);']);
end

%%
switch options
    case 'none'
        Rep = 1;
    case 'flipH'
        Rep = 2;
    case 'clipUL'
        T = 6;
        Rep = T^2;
end

N = length(im_lists);
train_ratio = 0.8;
for t = 1 : size(index, 1)
    fprintf('the %d-th iteration\n',t);

    srocc = zeros(length(layer_names), 3, 5);
    for k = 1:length(layer_names)
        layer_name = layer_names{k};
        im_index = index(t,1:ceil(train_ratio*size(index,2))); 
        I = buffer(im_index,5);
        for i = 1:5
            train_im_index = I; 
            train_im_index(i,:) = []; 
            train_im_index(train_im_index==0) = [];
            test_im_index = I(i,:);   
            test_im_index(test_im_index==0) = [];
            train_im_index = cell2mat(arrayfun(@(i)find(ref_ids==train_im_index(i))',...
                1:length(train_im_index),'UniformOutput',false));
            test_im_index = cell2mat(arrayfun(@(i)find(ref_ids==test_im_index(i))',...
                1:length(test_im_index),'UniformOutput',false));
            train_labels = subjective_scores(train_im_index); %#ok<NASGU>
            test_labels = subjective_scores(test_im_index);

            p = 10; % number of components;
            eval(['[~,~,~,~,beta1] = plsregress(feature1.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
            eval(['predict_statistics1 = [ones(length(test_im_index(:)),1) feature1.' layer_name '(test_im_index, :)]*beta1;']);
            eval(['[~,~,~,~,beta2] = plsregress(feature2.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
            eval(['predict_statistics2 = [ones(length(test_im_index(:)),1) feature2.' layer_name '(test_im_index, :)]*beta2;']);
            eval(['[~,~,~,~,beta3] = plsregress(feature3.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
            eval(['predict_statistics3 = [ones(length(test_im_index(:)),1) feature3.' layer_name '(test_im_index, :)]*beta3;']);
        %     
            predict_statistics = [predict_statistics1 predict_statistics2  predict_statistics3];

            srocc(k,:,i) = corr(predict_statistics, test_labels, 'type', 'Spearman');
        end
    end

    train_im_index = index(t,1:ceil(train_ratio*size(index,2)));    
    train_im_index = cell2mat(arrayfun(@(i)find(ref_ids==train_im_index(i))',...
        1:length(train_im_index),'UniformOutput',false));
    test_im_index = index(t,1+ceil(train_ratio*size(index,2)):size(index,2));
    test_im_index = cell2mat(arrayfun(@(i)find(ref_ids==test_im_index(i))',...
        1:length(test_im_index),'UniformOutput',false));
    train_labels = subjective_scores(train_im_index);
    test_labels = subjective_scores(test_im_index);

    Avesrocc = mean(srocc,3);
    [bestK, bestS] = find(Avesrocc==max(Avesrocc(:)),1,'first'); %

    layer_name = layer_names{bestK};  
    p = 10; % number of components;
    clear predict_statistics
    eval(['[~,~,~,~,beta1] = plsregress(feature1.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
    eval(['predict_statistics(:, 1) = [ones(length(test_im_index(:)),1) feature1.' layer_name '(test_im_index, :)]*beta1;']);
    eval(['[~,~,~,~,beta2] = plsregress(feature2.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
    eval(['predict_statistics(:, 2) = [ones(length(test_im_index(:)),1) feature2.' layer_name '(test_im_index, :)]*beta2;']);
    eval(['[~,~,~,~,beta3] = plsregress(feature3.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
    eval(['predict_statistics(:, 3) = [ones(length(test_im_index(:)),1) feature3.' layer_name '(test_im_index, :)]*beta3;']);
    w = Avesrocc(bestK,:)'/sum(Avesrocc(bestK,:));
    objective_scores = predict_statistics*w;
    results.srocc(t) = corr(objective_scores, test_labels, 'type', 'Spearman');
    results.plcc(t) = corr(objective_scores, test_labels);
    results.krocc(t) = corr(objective_scores, test_labels, 'type', 'Kendall');
    results.RMSE(t) = sqrt(mean((objective_scores-test_labels).^2));
    results.OR(t) = mean(abs(objective_scores-test_labels)>2*subjective_scoresSTD(test_im_index));
    median(results.srocc)
end
save(save_path);
