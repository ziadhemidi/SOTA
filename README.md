# Generalized_INR  
The pytorch implementation of our paper ["Generalized Implicit Neural Representation for MRI Parallel Imaging Reconstruction"](https://arxiv.org/abs/2309.06067)

## 1. Requirements  
python = 3.8  
torch = 1.8.1  
numpy = 1.21.1  

## 2. Dataset Preparation 
Please prepare your dataset in the following structure for easy use of this repository:  
```
your_data_path
	└── fastMRI_brain
    		├── train
        		├── xxx.npz
			├── ...
			└── ...
    		├── val
        		├── xxx.npz
			├── ...
			└── ...
    		└── test
        		├── xxx.npz
			├── ...
			└── ...
```
We recommend you to write your own `dataloader` in `data.py`.


## 3. Run

**GRAPPA:**  
To run GRAPPA, you should firstly specify the file path in `./GRAPPA/run.py`:
```
python GRAPPA/run.py
```

**RAKI:**  
To run RAKI, you should firstly specify the configurations in `./RAKI/run.py`:
```
python RAKI/run.py
```

**rRAKI:**  
To run rRAKI, you should firstly specify the configurations in `./rRAKI/run.py`:
```
python rRAKI/run.py
```

**IMJENSE:**  
To run IMJENSE, you should firstly specify the configurations in `./IMJENSE/run.py`:
```
python IMJENSE/run.py
```

**ReconFormer:**  
To run ReconFormer, you should firstly specify the configurations in `./ReconFormer/run.py`:
```
python ReconFormer/run.py
```

**Generalized_INR:**  
To run our model, you should firstly specify the configurations in `./multi_scale_recon/configs.py`:
```
python multi_scale_recon/run.py
```

## 4. Citation  
If you find this repository helpful in your work, please cite:
```bash
@article{li2024implicitneuralrepresentationmri,
         title={Implicit Neural Representation for MRI Parallel Imaging Reconstruction}, 
         author={Hao Li and Yusheng Zhou and Jianan Liu and Xiling Liu and Tao Huang and Zhihan Lv and Weidong Cai},
         year={2024},
         eprint={2309.06067},
         archivePrefix={arXiv},
         primaryClass={eess.IV},
         url={https://arxiv.org/abs/2309.06067}, 
}
```

## 5. Reference
This repository refers to:  

1. Generalized Autocalibrating Partially Parallel Acquisitions (GRAPPA) [[code](https://github.com/mckib2/pygrappa)] and [[paper](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.10171)]  

2. Scan‐specific Robust Artificial‐Neural‐Networks for K‐space Interpolation (RAKI) Reconstruction: Database‐free Deep Learning for Fast Imaging [[code](https://github.com/zczam/RAKI)] and [[paper](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.27420)]  

3. Residual RAKI: A Hybrid Linear and Non-linear Approach for Scan-specific K-space Deep Learning [[code](https://github.com/zczam/rRAKI)] and [[paper](https://doi.org/10.1016/j.neuroimage.2022.119248)]

4. IMJENSE: Scan-Specific Implicit Representation for Joint Coil Sensitivity and Image Estimation in Parallel MRI [[code](https://github.com/AMRI-Lab/IMJENSE)] and [[paper](https://ieeexplore.ieee.org/document/10356136)]  

5. ReconFormer: Accelerated MRI Reconstruction Using Recurrent Transformer [[code](https://github.com/guopengf/ReconFormer)] and [[paper](https://ieeexplore.ieee.org/document/10251064)]  


