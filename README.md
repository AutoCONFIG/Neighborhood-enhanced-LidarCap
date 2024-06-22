# Download dataset

1. Download LiDARHuman26m and wight file from link:
2. Download Cimi4D and HSC4D from link: https://smpl.is.tue.mpg.de/

If you want to extract 3BN and 3SN from your own data, refer to `sample/noise_select`.

# 2. Build Environment
```
conda create -n lidar_human python=3.7
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"（或者下载github然后pip install pointnet2_ops_lib/.）
pip install wandb
pip install h5py
pip install tqdm
pip install scipy
pip install opencv-python
pip install pyransac3d
pip install yacs
pip install plyfile
pip install scikit-image
pip install joblib
pip install chumpy
```



# Train or Eval
1. Modify `configs/base.yaml`to set `dataset_path` to the path where your dataset is located.
2. Update the relevant information for `wandb` in `main.py`
3. Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`,`J_regressor_extra.npy` from link: https://smpl.is.tue.mpg.de/
and put it in `unit` directory.

### Train 
```
CUDA_VISIBLE_DEVICES=x python main.py --debug
```

### Eval
```
CUDA_VISIBLE_DEVICES=x python main.py --ckpt_path same_data --debug --eval --eval_bs=1
```

# Citation
```@inproceedings{zhang2024neighborhood,
  title={Neighborhood-Enhanced 3D Human Pose Estimation with Monocular LiDAR in Long-Range Outdoor Scenes},
  author={Zhang, Jingyi and Mao, Qihong and Hu, Guosheng and Shen, Siqi and Wang, Cheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7169--7177},
  year={2024}
}