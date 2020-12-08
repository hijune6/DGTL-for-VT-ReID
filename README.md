# Strong but simple baseline with dual-granularity triplet loss for VT-ReID
Pytorch code for "Strong but Simple Baseline with Dual-Granularity Triplet Loss for Visible-Thermal Person Re-Identification"[(arxiv)](https://arxiv.org/abs/2008.06223).

### Highlights
- Our proposed dual-granularity triplet loss well organizes the sample-based triplet loss and center-based triplet loss in a hierarchical fine to coarse granularity manner, just with some simple configurations of typical operations, such as pooling and batch normalization.

- Experiments on RegDB and SYSU-MM01 datasets show that our DGTL can improve the VT-ReID performance with only the global features by large margins, which can be a strong VT-ReID baseline to boost the future research with high quality. 

### Results
       

Dataset| Rank1  | mAP | Rank1  | mAP 
 ---- | ----- | ------  |----- | ------  |
       | V->T  |        | T->V  |        
 RegDB | 83.92% | 73.78% | 81.59% | 71.65
       | all search  |        | indoor search  |        
 SYSU-MM01  | 57.34% | 55.13%  | 63.11% | 69.20%
 

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

**Parameters**: More parameters can be found in the script and code.

### 4. Citation

Please kindly cite the following papers in your publications if it helps your research:
```

```
