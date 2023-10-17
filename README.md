# DCGAN Implementation

The convolutional network using U-Net architecture is implemented in this project, for a Lung X-ray dataset with masks(ground truth) and its respective x-ray iamges. The results on the test dataset is elaborated below.

The implementation was done on Google colab. Images were downloaded on to google drive and subsequently used in the notebook. 

Refer  ``` training_code.ipynb``` for codes related to training the model, and plotting results, ```dataloader.py``` for data preprocessing steps, ```modelDCGAN.py``` for U-net classes and its architecture.




## Install Requirements:
For installing the requirements for this software, please run the following: 

 ```
 pip install -r requirements.txt
 ```
  

## Dataset
https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels/data