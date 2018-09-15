%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     Cross Database Experiments    %%%%%
%%%%%     SFA (Li et al. TMM 2018)      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% written by Dingquan Li
% dingquanli@pku.edu.cn
% IDM, SMS, PKU
% Last update: Sept. 15, 2018

clear;clc

caffe_path = '/home/ldq/caffe/matlab/'; % point to the caffe path
addpath(genpath(caffe_path)); 

%%
databases = {'LIVE','TID2008','TID2013','MLIVE','MLIVEblurjpeg','BID','CLIVE'};
for r = 1:length(databases)
    traindatabase = databases{r};
    %%
    load(['./results/' traindatabase '-reproduce']); %

    srocc = zeros(length(layer_names), 3, 5);
    for k = 1:length(layer_names)
        layer_name = layer_names{k};
        im_index = index(t, :); 
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
            eval(['[~,~,~,~,beta1k{k}] = plsregress(feature1.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
            eval(['predict_statistics1 = [ones(length(test_im_index(:)),1) feature1.' layer_name '(test_im_index, :)]*beta1k{k};']);
            eval(['[~,~,~,~,beta2k{k}] = plsregress(feature2.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
            eval(['predict_statistics2 = [ones(length(test_im_index(:)),1) feature2.' layer_name '(test_im_index, :)]*beta2k{k};']);
            eval(['[~,~,~,~,beta3k{k}] = plsregress(feature3.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
            eval(['predict_statistics3 = [ones(length(test_im_index(:)),1) feature3.' layer_name '(test_im_index, :)]*beta3k{k};']);
        %     
            predict_statistics = [predict_statistics1 predict_statistics2  predict_statistics3];

            srocc(k,:,i) = corr(predict_statistics, test_labels, 'type', 'Spearman');
        end
    end
    Avesrocc = mean(srocc,3);
    [bestK, bestS] = find(Avesrocc==max(Avesrocc(:)),1,'first'); %
    w = Avesrocc(bestK,:)'/sum(Avesrocc(bestK,:));
    bestLayer = layer_names{bestK};
    beta1 = beta1k{bestK};
    beta2 = beta2k{bestK};
    beta3 = beta3k{bestK};


    Cross.bestLayer{r} = bestLayer;

    train_im_index = 1:size(index,2); 
    train_im_index = cell2mat(arrayfun(@(i)find(ref_ids==train_im_index(i))',...
            1:length(train_im_index),'UniformOutput',false));
    train_labels = subjective_scores(train_im_index); 
    layer_name = bestLayer;  
    eval(['[~,~,~,~,beta.' layer_name '1] = plsregress(feature1.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
    eval(['[~,~,~,~,beta.' layer_name '2] = plsregress(feature2.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
    eval(['[~,~,~,~,beta.' layer_name '3] = plsregress(feature3.' layer_name '(train_im_index(:)+[0:N:(Rep-1)*N],:),repmat(train_labels,Rep,1),p);']);
    Cross.beta{r} = beta;
    % w
    %%
    for c = 1:length(databases)
        testdatabase = databases{c};
        load(['./results/' testdatabase '-reproduce']); %

        test_im_index = 1:size(index,2);
        test_im_index = cell2mat(arrayfun(@(i)find(ref_ids==test_im_index(i))',...
            1:length(test_im_index),'UniformOutput',false));
        test_labels = subjective_scores(test_im_index);
        layer_name = bestLayer;  
        eval(['predict_statistics1 = [ones(length(test_im_index(:)),1) feature1.' layer_name '(test_im_index, :)]*beta.' layer_name '1;']);
        eval(['predict_statistics2 = [ones(length(test_im_index(:)),1) feature2.' layer_name '(test_im_index, :)]*beta.' layer_name '2;']);
        eval(['predict_statistics3 = [ones(length(test_im_index(:)),1) feature3.' layer_name '(test_im_index, :)]*beta.' layer_name '3;']);

        w1 = 1/3; w2 = 1/3; w3 = 1/3;
        predict_statistics = w1*predict_statistics1 + w2*predict_statistics2 + w3*predict_statistics3;
    Cross.objective_scores{r,c} = predict_statistics;

    [Cross.SROCC(r,c),Cross.KROCC(r,c),Cross.PLCC(r,c),Cross.OR(r,c),Cross.RMSE(r,c),Cross.mapped_scores{r,c}] = ...
        nonlinearfitting(predict_statistics, test_labels, subjective_scoresSTD(test_im_index));
    Cross.SROCC
    end
end
% save('./results/Cross','Cross');