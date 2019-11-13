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
```bash
python SFA.py --database=CLIVE
```
### Remark
The reported results in the paper are obtained by Caffe framework. Here we only provide a PyTorch implementation of the method.
 
TODO:
- Add more information and details.
- etc.

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
