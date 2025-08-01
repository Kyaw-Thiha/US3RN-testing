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



## Train
Before training, ensure that in `get_patch_training_set` of `data.py`, the correct folder is chosen.

### Train (from 0 Epoch)
```python
python main.py --mode train --upscale_factor 2 --ChDim 81 --lr 0.01
```

### Train (from 10 Epoch)
```python
python main.py --mode train --upscale_factor 2 --ChDim 81 --lr 0.01 --nEpochs 10
```

### Train (from 10 to 50 Epoch)
```python
python main.py --mode train --upscale_factor 2 --ChDim 81 --lr 0.01 --nEpochs 10 --endEpochs 50
```

### Train (Custom Step)
```python
python main.py --mode train --upscale_factor 2 --ChDim 81 --lr 0.01 --save_step 1 --nEpochs 10 --endEpochs 20
```
-----

## Test
Before testing, ensure that in `get_testing_set` of `data.py`, the correct folder is chosen.

```python
python main.py --mode test --upscale_factor 2 --nEpochs 150
```

### Batch Test
```python
python main.py --mode batch_test --upscale_factor 2
```

### For logging out errors
```python
python main.py --mode test --nEpochs 152 > test_error.log 2>&1
```

-----
## Dataset Preparation
1. Create a folder called `data` in the root directory.
2. Create a folder for each of the dataset, and add either `.mat` or `.npy` file into there.
Example: `/data/Indian_pines/Indian_pines.mat`. Note that file name does not matter.
3. Go to `preprocess/test.py` and `preprocess/train.py` and change the data source if needed. 
4. Ensure that the correct corresponding folder is chosen in `data.py`
5. Run either or both of the following command from the root_directory.
```bash
python -m preprocess.train
```


```bash
python -m preprocess.test
```
-----

## Analytics
By default, all training logs are saved in
- `logs/csv/epoch_logs.csv` and `logs/csv/batch_logs.csv`
- `logs/train_logs/train_n_.log` for all printed outputs
- Tensorboard logs at `tb_logger` folder

### Plotting logs
```python
python -m analytics.plot_train
```

```python
python -m analytics.plot_test
```

### Importing logs from .log file into .csv file
Make sure to edit the log and csv file paths first in the python file before running the following command.

```python
python -m analytics.import_logs_to_csv
```

## Setting up on Remote Server
1. `chmod +x setup.sh`
2. `./setup.sh`
3. Run the training script

