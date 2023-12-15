# [Re] Denoising Diffusion Restoration Models 

**Authors**: Dalim Wahby, Philipp Ahrendt, Iga Pawlak

Implemented as a final project for [Deep Learning, advanced](https://www.kth.se/student/kurser/kurs/DD2412?l=en) course given at KTH Stockholm, academic year 2023/2024.

---
A reproduction of Denoising Diffusion Restoration Models ([DOI](https://arxiv.org/abs/2201.11793)) for solving linear inverse problems in imaging using Diffusion Models. 

We use the pre-trained models from [https://github.com/pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) and create a class inheriting from their ´Diffusion´ class and modify the ´denoise´ function as well as ´denoising_step´ 
with implementations of modifications described in DDRM. 

### File structure 


