# US3RN-Pytorch
The code is for the work:

```
@article{ma2021deep,
  title={Deep Unfolding Network for Spatiospectral Image Super-Resolution},
  author={Qing Ma, Junjun Jiang, Xianming Liu, and Jiayi Ma},
  journal={IEEE Transactions on Computational Imaging},
  volume={},
  number={},
  pages={},
  year={2022},
}
```



## Requirements

``` python
pytorch == 1.6.1

```

### Dataset

To train and test on CAVE data set, you must first download the CAVE data set form http://www.cs.columbia.edu/CAVE/databases/multispectral/. Put all the training images and test images in their respective folders. You can also download the processed data from https://drive.google.com/drive/folders/1lwsNkmDFW81PvRGPWWBh-5wQDtF8XgQ5?usp=sharing 

## Train

### Train (from 0 Epoch)
```python
python main.py --mode train --upscale_factor 2 --ChDim 81 --lr 0.01
```

### Train (from 10 Epoch)
```python
python main.py --mode train --upscale_factor 2 --ChDim 81 --lr 0.01 --nEpochs 10
```


## Test

```python
python main.py --mode test --nEpochs 150
```


### For logging out errors
```python
python main.py --mode test --nEpochs 152 > test_error.log 2>&1
```

