PyTorch 1.3 implementation of the following papers:

- Li, Dingquan, Tingting Jiang, and Ming Jiang. "[Exploiting High-Level Semantics for No-Reference Image Quality Assessment of Realistic Blur Images](https://dl.acm.org/citation.cfm?id=3123266.3123322)." Proceedings of the 2017 ACM on Multimedia Conference. ACM, 2017.

## Requirements
```bash
conda create -n reproducibleresearch pip python=3.6
source activate reproducibleresearch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
source deactive
```
- Python 3.6.8
- PyTorch 1.3.0



## Experiments
Before running the following commands, please copy the corresponding data to `data/`, change `args.im_dir` to your path, and download the transferred models provided in [ruotianluo/pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet): [resnet50-caffe.pth](https://drive.google.com/open?id=0B7fNdx_jAqhtbllXbWxMVEdZclE) and [resnet50.pth](https://drive.google.com/uc?export=download&id=0B7fNdx_jAqhtam1MSTNSYXVYZ2s). For people who cannot access Google drive or the download speed is slow, you can download my backup [here](https://pan.baidu.com/s/1_ZuqGOL2TjPGDUDJfk_hKg) (password: il68).
```bash
python SFAwithCaffemodelweights.py --database=CLIVE --model_path=resnet50-caffe.pth
# Output: [ 0.80362151  0.61258695  0.82201323 11.68561182  0.8583691 ]
```

```bash
python SFA.py --database=CLIVE --model_path=resnet50.pth
# Output: [ 0.80366405  0.61277194  0.8219469  11.68904924  0.8583691 ]
```

### Remark
**The reported results in the paper are obtained by Caffe framework**. 
If you want to compare the SFA model in your research, it would be better to use the original Caffe+MATLAB version for training and testing in the same settings as yours.
Here we only provide a PyTorch implementation of the method as illustration for researchers who use PyTorch framework and want to improve SFA model. 
You should know the major differences between the above two versions are the pre-trained model weights and the corresponding pre-processing step.

The provided files correspond to the conference version with minor difference, and the obtained results with PyTorch is slightly worse than the results with Caffe.

 
TODO:
- Add adaptive layer selection procedure used in the journal version.
- Replace line 35 in `SFAwithCaffemodelweights.py` with the mean (ImageNet) image subtracted.
- Add more information and details, etc.


## Citation

Please cite our papers if it helps your research:

<pre><code>@arcticle{li2019which,
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

<pre><code>@inproceedings{li2017exploiting,
  title={Exploiting High-Level Semantics for No-Reference Image Quality Assessment of Realistic Blur Images},
  author={Li, Dingquan and Jiang, Tingting and Jiang, Ming},
  booktitle={Proceedings of the 2017 ACM on Multimedia Conference},
  pages={378--386},
  year={2017},
  organization={ACM}
}</code></pre>

## Contact
Dingquan Li, dingquanli AT pku DOT edu DOT cn.
