# inpainting-FSIPA
## Requirements
Python >= 3.5
PyTorch >= 1.0.0
Opencv2 ==3.4.1
Scipy == 1.1.0
Numpy == 1.14.3
Scikit-image (skimage) == 0.13.1
visdom == 0.1.8.9
lpips ==0.1.4
## Dataset Preparation
We use [Places2](http://places2.csail.mit.edu/download.html), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Paris Street-View datasets](https://github.com/pathak22/context-encoder). Except Places2, train a model on the full dataset, download datasets from official websites.

Our model is trained on the irregular mask dataset provided by [Liu et al](https://arxiv.org/abs/1804.07723). You can download publically available Irregular Mask Dataset from their [website](https://nv-adlr.github.io/publication/partialconv-inpainting).

For Structure image of datasets, we follow the [structure flow](https://github.com/RenYurui/StructureFlow) and utlize the [RTV smooth method](http://www.cse.cuhk.edu.hk/~leojia/projects/texturesep/).Run generation function [data/Matlab/generate_structre_images.m](https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/blob/master/data/Matlab/generate_structure_images.m) in your matlab. For example, if you want to generate smooth images for Places2, you can run the following code:
```
generate_structure_images("path to Places2 dataset root", "path to output folder");
```
## Running the program
To perform training or testing, use 
```
python run.py
```
There are several arguments that can be used, which are
```
--data_root +str #where to get the images for training/testing
--edge_root+str #where to get the smooth images for training/testing
--mask_root +str #where to get the masks for training/testing
--model_save_path +str #where to save the model during training
--result_save_path +str #where to save the inpainting results during testing
--g_path +str #the pretrained generator to use during training/testing
--d_path +str #the pretrained discriminator to use during training/testing
--target_size +int #the size of images and masks
--mask_mode +int #which kind of mask to be used, 0 for external masks with random order, 1 for randomly generated masks, 2 for external masks with fixed order
--batch_size +int #the size of mini-batch for training
--n_threads +int
--gpu_id +int #which gpu to use
--finetune #to finetune the model during training
--test #test the model
```
For example, to train the network using gpu 1, with pretrained models
```
python run.py --data_root data --mask_root mask --g_path checkpoints/g_10000.pth --d_path checkpoints/d_10000.pth --batch_size 6 --gpu 1
```
to test the network
```
python run.py --data_root data --mask_root mask --g_path checkpoints/g_10000.pth --test --mask_mode 2
```
to evalue the results
```
python quality_test.py
```
Before perform evalue , to t two arguments firstly:
```
--real_path +str #where to get the groundtruth image
--fake_path+str #where to get the generated image
```
## Pretrained model
All the descriptions below are under the assumption that parameters are consistent with our paper,and we will up pre-trained model later.
We strongly encourage the users to retrain the models if they are used for academic purpose, to ensure fair comparisons (which has been always desired). Achieving a good performance using the current version of code should not be difficult

## Training procedure
To fully exploit the performance of the network, we suggest to use the following training procedure, in specific
1. Train the network, i.e. use the command
```
python run.py
```
2. Finetune the network, i.e. use the command
```
python run.py --finetune --g_path path-to-trained-generator --d_path path-to-trained-discriminator
```
3. Test the model
```
python run.py --test
```
4. Evalue the model results
 ```
python quality_test.py

```
