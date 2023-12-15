# [Re] Denoising Diffusion Restoration Models 

**Authors**: Philipp Ahrendt, Dalim Wahby, Iga Pawlak

Implemented as a final project for [Deep Learning, advanced](https://www.kth.se/student/kurser/kurs/DD2412?l=en) course given at KTH Stockholm, academic year 2023/2024.

---
A reproduction of Denoising Diffusion Restoration Models ([DOI](https://arxiv.org/abs/2201.11793)) for solving linear inverse problems in imaging using Diffusion Models. 

## Abstract
In this work we aim to re-implement the Denoising Diffusion Restoration Models (DDRM), on selected tasks, such as 1) denosing, 2) deburring, and 3) super-resolution. Furthermore, we investigate the performance of a less-informed prior of different models. Additionally, we introduce a novel metric, that accesses the quality of a generated image, called the GIQA score. Our experiments yield similar results to those of the original paper and we found that a less-informed prior does not necessarily impact the performance of a model, thanks to a multitude of features being incorporated training data. However, for specialized tasks, the prior should be chosen carefully, since  the generated images might lack realism. Furthermore, we found that the GIQA score returns realistic values, with the generated image having a score between the degraded and original score, implicating that it is a relevant measure for the quality of generated images. In conclusion, this work contributes to the understanding of DDRM, by re-implementing the method and clarifying yet unclear passages of the original paper.

### File structure

``````
├── DDRM
│   ├── data
│   │   ├── ddrm-exp-datasets-main
│   │   │   ├── ood
│   │   │   ├── ood_bedroom
│   │   │   ├── ood_celeba
│   │   │   └── ood_church_outdoor
│   │   └── image_net_1k
│   ├── experiments_image_net_data
│   │   ├── evaluation_deblurring_image_net.ipynb
│   │   ├── evaluation_denoising_image_net.ipynb
│   │   ├── evaluation_noiseless_deblurring_iamge_net.ipynb
│   │   └── evaluation_super_resolution_image_net.ipynb
│   ├── experiments_ood_data
│   │   ├── evaluation_deblurring.ipynb
│   │   ├── evaluation_denoising.ipynb
│   │   ├── evaluation_noiseless_deblurring.ipynb
│   │   ├── evaluation_super_resolution_noise.ipynb
│   │   └── evaluation_super_resolution_noiseless.ipynb
│   └── utils
│       ├── data_utils.py
│       ├── ddrm.py
│       └── degradation.py
└── README.md
``````

## Data
The data used for our expiments can be found here:
- Minimum set of images need to run on the DDRM paper: [OOD Data](https://github.com/jiamings/ddrm-exp-datasets)
- Huggingface's [ImageNet 1k](https://huggingface.co/datasets/imagenet-1k)


## Acknowledgements
Original paper: Kawar et al. (2022) [Denoising Diffusion Restoration Models](https://arxiv.org/abs/2201.11793)

We use the pre-trained models from [Pytorch Diffusion](https://github.com/pesser/pytorch_diffusion) and create a class inheriting from their ´Diffusion´ class and modify the ´denoise´ function as well as ´denoising_step´ with implementations of modifications described in DDRM.