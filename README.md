# Which Has Better Visual Quality: The Clear Blue Sky or a Blurry Animal?
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](License)

## Description
SFA code for the following papers:

- Li, Dingquan, Tingting Jiang, Weisi Lin, and Ming Jiang. "[Which Has Better Visual Quality: The Clear Blue Sky or a Blurry Animal?](https://ieeexplore.ieee.org/document/8489929)." IEEE Transactions on Multimedia, vol. 21, no. 5, pp. 1221-1234, May 2019.
- Li, Dingquan, Tingting Jiang, and Ming Jiang. "[Exploiting High-Level Semantics for No-Reference Image Quality Assessment of Realistic Blur Images](https://dl.acm.org/citation.cfm?id=3123266.3123322)." Proceedings of the 2017 ACM on Multimedia Conference. ACM, 2017.

## Requirement
Framework: [Caffe](https://github.com/BVLC/caffe/) 1.0 + [MATLAB](https://www.mathworks.com/products/matlab.html) 2016b Interface

The PLSR model uesd in the test code is trained on [LIVE](http://live.ece.utexas.edu/research/Quality/subjective.htm) gblur images with DMOS (the larger the worse).

Download the ResNet-50-model.caffemodel from https://github.com/KaimingHe/deep-residual-networks and paste it into the directory "models/" before using the code!

**New!** We provide the [PyTorch](https://pytorch.org) implementation of the method in [SFA-pytorch](./SFA-pytorch/)

## Note for training
All we need to train is a PLSR model, where the training function is `plsregress` in [MATLAB](https://www.mathworks.com/products/matlab.html). The features are extracted from the DCNN models pre-trained on the image classification task.

Update: remember to change the value of "im_dir" and "im_lists" in data info.

## Note for datasets
You can download the datasets used in the papers from their owners for research purpose. If you have difficulty, you can refer to [my sharing on BaiduNetDisk](https://pan.baidu.com/s/10rGDziwuQl0fjwDHlhX1dA) with password `b6sv`. We only consider the blur related images in this work.

## Citation

Please cite our papers if it helps your research:

<pre><code>@arcticle{li2018which,
  title={Which Has Better Visual Quality: The Clear Blue Sky or a Blurry Animal?},
  author={Li, Dingquan and Jiang, Tingting and Lin, Weisi and Jiang, Ming},
  booktitle={IEEE Transactions on Multimedia},
  volume={21}, 
  number={5}, 
  pages={1221-1234},  
  month={May},
  year={2019}, 
  doi={10.1109/TMM.2018.2875354}
}</code></pre>

[[Paper](https://ieeexplore.ieee.org/document/8489929)] 

<pre><code>@inproceedings{li2017exploiting,
  title={Exploiting High-Level Semantics for No-Reference Image Quality Assessment of Realistic Blur Images},
  author={Li, Dingquan and Jiang, Tingting and Jiang, Ming},
  booktitle={Proceedings of the 2017 ACM on Multimedia Conference},
  pages={378--386},
  year={2017},
  organization={ACM}
}</code></pre>

[[Paper](https://www.researchgate.net/profile/Dingquan_Li3/publication/320541334_Exploiting_High-Level_Semantics_for_No-Reference_Image_Quality_Assessment_of_Realistic_Blur_Images/links/5a0c14cba6fdccc69edaa035/Exploiting-High-Level-Semantics-for-No-Reference-Image-Quality-Assessment-of-Realistic-Blur-Images.pdf)] 
[[Poster](./acmmm17_poster-updated.pdf)]

## Contact
Dingquan Li, dingquanli AT pku DOT edu DOT cn.
