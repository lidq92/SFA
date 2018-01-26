# Exploiting High-Level Semantics for No-Reference Image Quality Assessment of Realistic Blur Images

## Description
SFA-PLSR (Test code) for the following paper:
Li, Dingquan, Tingting Jiang, and Ming Jiang. "Exploiting High-Level Semantics for No-Reference Image Quality Assessment of Realistic Blur Images." Proceedings of the 2017 ACM on Multimedia Conference. ACM, 2017.

## Requirement
Framework: Caffe + MATLAB 2016b Interface

The PLSR model uesd in the test code is trained on LIVE gblur images. 

Download the ResNet-50-model.caffemodel from https://github.com/KaimingHe/deep-residual-networks and paste it into the directory "models/" before using the code!

## Note for training
All we need to train is a PLSR model, where the training function is plsregress.m in MATLAB. The features are extracted from the DCNN models pre-trained on the image classification task.

## Citation

Please cite our paper if it helps your research:

<pre><code>@inproceedings{li2017exploiting,
  title={Exploiting High-Level Semantics for No-Reference Image Quality Assessment of Realistic Blur Images},
  author={Li, Dingquan and Jiang, Tingting and Jiang, Ming},
  booktitle={Proceedings of the 2017 ACM on Multimedia Conference},
  pages={378--386},
  year={2017},
  organization={ACM}
}</code></pre>

[[Paper](https://www.researchgate.net/profile/Dingquan_Li3/publication/320541334_Exploiting_High-Level_Semantics_for_No-Reference_Image_Quality_Assessment_of_Realistic_Blur_Images/links/5a0c14cba6fdccc69edaa035/Exploiting-High-Level-Semantics-for-No-Reference-Image-Quality-Assessment-of-Realistic-Blur-Images.pdf)] 
[[Poster](./acmmm17_poster.pdf)]

## Contact
Dingquan Li, dingquanli@pku.edu.cn.

## License
MIT License