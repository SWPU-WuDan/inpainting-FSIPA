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
We use Places2, CelebA and Paris Street-View datasets. To train a model on the full dataset, download datasets from official websites.

Our model is trained on the irregular mask dataset provided by Liu et al. You can download publically available Irregular Mask Dataset from their website.

For Structure image of datasets, we follow the structure flow and utlize the RTV smooth method.Run generation function data/Matlab/generate_structre_images.m in your matlab. For example, if you want to generate smooth images for Places2, you can run the following code:

generate_structure_images("path to Places2 dataset root", "path to output folder");

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
