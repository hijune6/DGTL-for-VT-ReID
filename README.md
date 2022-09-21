# Strong but simple baseline with dual-granularity triplet loss for VT-ReID
Pytorch code for "Strong but Simple Baseline with Dual-Granularity Triplet Loss for Visible-Thermal Person Re-Identification"[(arxiv)](https://arxiv.org/abs/2012.05010).

### Highlights
- Our proposed dual-granularity triplet loss well organizes the sample-based triplet loss and center-based triplet loss in a hierarchical fine to coarse granularity manner, just with some simple configurations of typical operations, such as pooling and batch normalization.

- Experiments on RegDB and SYSU-MM01 datasets show that our DGTL can improve the VT-ReID performance with only the global features by large margins, which can be a strong VT-ReID baseline to boost the future research with high quality. 

### Results
       

|Dataset| Rank1  | mAP | | Rank1  | mAP |
| :-----: | -----: | :------ |-|-----: | :------ |
|      |   visible to|thermal     | |   thermal to|visible  |
| RegDB | 83.92% | 73.78% | |  81.59% | 71.65%  |
|      |   all|search      | |  indoor|serach     |
| SYSU-MM01  | 57.34% | 55.13%  | | 63.11% | 69.20% |
 

### Usage
Our code extends the pytorch implementation of Cross-Modal-Re-ID-baseline in [Github](https://github.com/mangye16/Cross-Modal-Re-ID-baseline). Please refer to the offical repo for details of data preparation.

### Training
Train a model for RegDB by
```bash
python train.py --dataset regdb --lr 0.1 --gpu 0 --bpool max --cpool max --hcloss HcTri
```

Train a model for SYSU-MM01 by
```bash
python train.py --dataset sysu --lr 0.1 --batch-size 6 --num_pos 8 --gpu 1 --bpool avg --cpool max --hcloss HcTri --margin_hc 0.5
```

**Parameters**: More parameters can be found in the manuscript and code.

### 4. Citation

Please kindly cite the following paper in your publications if it helps your research:
```
@article{liu2020parameter,
  title={Parameter sharing exploration and hetero-center triplet loss for visible-thermal person re-identification},
  author={Liu, Haijun and Tan, Xiaoheng and Zhou, Xichuan},
  journal={IEEE Transactions on Multimedia},
  volume={23},
  pages={4414--4425},
  year={2020},
  publisher={IEEE}
}
```
```
@article{liu2021strong,
  title={Strong but simple baseline with dual-granularity triplet loss for visible-thermal person re-identification},
  author={Liu, Haijun and Chai, Yanxia and Tan, Xiaoheng and Li, Dong and Zhou, Xichuan},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={653--657},
  year={2021},
  publisher={IEEE}
}
```
```
@article{Tan2022AFS,
  title={A Fourier-Based Semantic Augmentation for Visible-Thermal Person Re-Identification},
  author={Xiaoheng Tan and Yanxia Chai and Fenglei Chen and Haijun Liu},
  journal={IEEE Signal Processing Letters},
  year={2022},
  volume={29},
  pages={1684-1688}
}
```
