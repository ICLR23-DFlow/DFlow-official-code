# [ICLR 2023] DFlow: Learning to Synthesize Better Optical Flow Datasets via a Differentiable Pipeline ([Paper](https://openreview.net/pdf?id=5O2uzDusEN5))

<h4 align="center">Kwon Byung-Ki<sup>1</sup>, Nam Hyeon-Woo<sup>1</sup>, Ji-Yun Kim<sup>1</sup>, Tae-Hyun Oh<sup>1,2,3</sup></center>
<h4 align="center">1. Department of Electrical Engineering, POSTECH , 2. Graduate School of AI, POSTECH</center>
<h4 align="center">3. Institute for Convergence Research and Education in Advanced Technology, Yonsei University</center>


## Abstract
Comprehensive studies of synthetic optical flow datasets have attempted to reveal what properties lead to accuracy improvement in learning-based optical flow estimation. However, manually identifying and verifying the properties that contribute to accurate optical flow estimation require large-scale trial-and-error experiments with iteratively generating whole synthetic datasets and training on them, i.e., impractical. To address this challenge, we propose a differentiable optical flow data generation pipeline and a loss function to drive the pipeline, called DFlow. DFlow efficiently synthesizes a dataset effective for a target domain without the need for cumbersome try-and-errors. This favorable property is achieved by proposing an efficient dataset comparison method that uses neural networks to approximately
encode each dataset and compares the proxy networks instead of explicitly comparing datasets in a pairwise way. Our experiments show the competitive performance of our DFlow against the prior arts in pre-training. Furthermore, compared to competing datasets, DFlow achieves the best fine-tuning performance on the Sintel public benchmark with RAFT.

## Environment
- torch=2.0.1
- torchvision=0.10.0
- numpy=1.21.4
- kornia==0.7.0

## Rendered Datasets
You can download the our DFlow dataset and the pretrained model from the link below.
- DFlow dataset (https://drive.google.com/drive/folders/1_gu-N_Dfaywd-o-Ag9CMPoMfXurdLCUN?usp=sharing)
- Pretrained model (https://drive.google.com/drive/folders/1LLpxyGcgE9rUoWsZMdC07bdrV4QQBMEV?usp=sharing)

## Training from scratch
Download the DFlow dataset and place it in the "dataset" folder. Then, enter the "bash train_standard.sh" command
When training with the DFlow dataset, the data augmentation used during dataset creation is applied. Please refer to section B.1 in our paper for more details.

## Citation
If you find this work useful for your research, please cite: 
```
@inproceedings{byung2022dflow,
  title={DFlow: Learning to Synthesize Better Optical Flow Datasets via a Differentiable Pipeline},
  author={Byung-Ki, Kwon and Hyeon-Woo, Nam and Kim, Ji-Yun and Oh, Tae-Hyun},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2022}
}

```

## Acknowledgements
Part of the code is adapted from previous works:
- [RAFT](https://github.com/princeton-vl/RAFT)
- [Softmax Splatting](https://github.com/sniklaus/softmax-splatting)
