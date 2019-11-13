# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2019/11/13

from argparse import ArgumentParser
from PIL import Image
import os
import h5py
import numpy as np
import random
from torchvision import models
import torch.nn as nn
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize
from sklearn.cross_decomposition import PLSRegression
# from sklearn.svm import LinearSVR
from scipy import stats
import time


def default_loader(path, channel=3):
    if channel == 1:
        return Image.open(path).convert('L')
    else:
        assert (channel == 3)
        return Image.open(path).convert('RGB')  #


def OverlappingCropPatches(im, patch_size=224, stride=112):
    w, h = im.size
    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patch = normalize(patch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            patches = patches + (patch,)
    return torch.stack(patches)


class IQADataset(Dataset):
    def __init__(self, args, loader=default_loader):
        self.loader = loader
        self.less_memory = args.less_memory
        Info = h5py.File(args.data_info, 'r')
        self.mos = Info['subjective_scores'][0, :]
        self.mos_std = Info['subjective_scoresSTD'][0, :]
        self.im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(self.mos))]

        self.patches = []
        self.im = []
        self.label = []
        self.label_std = []
        for idx in range(len(self.im_names)):
            print("Preprocessing Image: {}".format(self.im_names[idx]))
            im = self.loader(os.path.join(args.im_dir, self.im_names[idx]))
            if self.less_memory:
                # If you have less memory, do the pre-processing later.
                self.im.append(im)
            else:
                patches = OverlappingCropPatches(im)
                self.patches.append(patches)  #
            self.label.append(self.mos[idx])
            self.label_std.append(self.mos_std[idx])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.less_memory:
            patches = OverlappingCropPatches(self.im[idx])
        else:
            patches = self.patches[idx]
        return patches


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])

    def forward(self, x):
        # features@: pool5
        x = self.features(x)

        return x.view(x.size(0), -1)


if __name__ == "__main__":
    parser = ArgumentParser(description='"SFA')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='BID', type=str,
                        help='database name (default: BID)')
    parser.add_argument('--less_memory', action='store_true',
                        help='flag whether to use less memory')
    parser.add_argument('--less_gpu_memory', action='store_true',
                        help='flag whether to use less GPU memory')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'LIVE':
        args.data_info = './data/LIVEinfo.mat'
        args.im_dir = '/media/ldq/Research/Data/databaserelease2/gblur/'

    if args.database == 'TID2013':
        args.data_info = './data/TID2013info.mat'
        args.im_dir = '/disk/ldq/Data/tid2013/distorted_images/'

    if args.database == 'TID2008':
        args.data_info = './data/TID2008info.mat'
        args.im_dir = '/media/ldq/Research/Data/tid2008/distorted_images/'

    if args.database == 'CLIVE':
        args.data_info = './data/CLIVEinfo.mat'
        args.im_dir = '/media/ldq/Research/Data/ChallengeDB_release/Images/'

    if args.database == 'BID':
        args.data_info = './data/BIDinfo.mat'
        args.im_dir = '/media/ldq/Research/Data/BID/ImageDatabase/'
        args.less_memory = True

    if args.database == 'MLIVE1':
        args.data_info = './data/MLIVE1info.mat'
        args.im_dir = '/media/ldq/Research/Data/LIVEmultidistortiondatabase/Part 2/blurnoise/'

    if args.database == 'MLIVE2':
        args.data_info = './data/MLIVE2info.mat'
        args.im_dir = '/media/ldq/Research/Data/LIVEmultidistortiondatabase/Part 1/blurjpeg/'

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    extractor = ResNet50().to(device)
    extractor.eval()
    dataset = IQADataset(args)
    X1, X2, X3 = [], [], []
    for i in range(len(dataset)):
        print('Extracting features of the {}-th image'.format(i))
        with torch.no_grad():
            if args.less_gpu_memory:
                # If you have less GPU memory, extract features of a batch of patches inseatd.
                # For example, set batch size = 1
                features = torch.zeros((len(dataset[i]), 2048))
                for j in range(len(dataset[i])):
                    features[j] = extractor(dataset[i][j:j+1].to(device))
            else:
                features = extractor(dataset[i].to(device))
            X1.append(torch.cat((torch.mean(features, dim=0),
                                 torch.std(features, dim=0))).to('cpu').numpy())
            X2.append(torch.cat((torch.mean(features, dim=0),
                                 torch.sqrt(torch.mean(torch.pow(features, 2), dim=0)),
                                 torch.pow(torch.mean(torch.pow(features, 3), dim=0), 1/3),
                                 torch.pow(torch.mean(torch.pow(features, 4), dim=0), 1/4))).to('cpu').numpy())
            features = features.to('cpu').numpy()
            X3.append(np.concatenate((np.min(features, axis=0),
                                      np.quantile(features, 0.25, axis=0),
                                      np.median(features, axis=0),
                                      np.quantile(features, 0.75, axis=0),
                                      np.max(features, axis=0))))

    Info = h5py.File(args.data_info, 'r')
    all_index = Info['index']
    ref_ids = Info['ref_ids'][0, :]
    criteria = np.zeros((all_index.shape[1], 5))
    train_ratio = 0.8
    for exp_id in range(all_index.shape[1]):
        print('EXP_ID: {}'.format(exp_id))
        index = all_index[:, exp_id]  # np.random.permutation(N)
        train_index, test_index = [], []
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in index[:int(train_ratio * len(index))]) else \
                test_index.append(i)

        X1_train = [X1[idx] for idx in train_index]
        X2_train = [X2[idx] for idx in train_index]
        X3_train = [X3[idx] for idx in train_index]
        X1_test = [X1[idx] for idx in test_index]
        X2_test = [X2[idx] for idx in test_index]
        X3_test = [X3[idx] for idx in test_index]
        y_train = [dataset.label[idx] for idx in train_index]
        y_test = [dataset.label[idx] for idx in test_index]
        y_test_std = [dataset.label_std[idx] for idx in test_index]

        start = time.time()
        # # SVR is slow.
        # regr1 = LinearSVR(random_state=0)
        # regr1.fit(X1_train, y_train)
        # y1_pred = regr1.predict(X1_test)
        # regr2 = LinearSVR(random_state=0)
        # regr2.fit(X2_train, y_train)
        # y2_pred = regr2.predict(X2_test)
        # regr3 = LinearSVR(random_state=0)
        # regr3.fit(X3_train, y_train)
        # y3_pred = regr3.predict(X3_test)
        pls10_1 = PLSRegression(n_components=10)
        pls10_1.fit(X1_train, y_train)
        y1_pred = pls10_1.predict(X1_test)
        pls10_2 = PLSRegression(n_components=10)
        pls10_2.fit(X2_train, y_train)
        y2_pred = pls10_2.predict(X2_test)
        pls10_3 = PLSRegression(n_components=10)
        pls10_3.fit(X3_train, y_train)
        y3_pred = pls10_3.predict(X3_test)
        stop = time.time()
        print("Time: {} seconds".format(stop-start))

        y_pred = (y1_pred + y2_pred + y3_pred) / 3

        y_pred = np.reshape(np.asarray(y_pred), (-1,))
        y_test = np.reshape(np.asarray(y_test), (-1,))
        y_test_std = np.reshape(np.asarray(y_test_std), (-1,))

        SROCC = stats.spearmanr(y_pred, y_test)[0]
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
        OR = (np.abs(y_pred-y_test) > 2 * y_test_std).mean()
        print('SROCC: {} KROCC: {} PLCC: {} RMSE: {} OR: {}%'.format(SROCC, KROCC, PLCC, RMSE, OR * 100))
        criteria[exp_id, :] = [SROCC, KROCC, PLCC, RMSE, OR * 100]
    print('Median results:')
    print(np.median(criteria, axis=0))
