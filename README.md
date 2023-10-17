# U-Net Implementation

The convolutional network using U-Net architecture is implemented in this project, for a Lung X-ray dataset with masks(ground truth) and its respective x-ray iamges. The results on the test dataset is elaborated below.

The implementation was done on Google colab. Images were downloaded on to google drive and subsequently used in the notebook. 

Refer  ``` training_code.ipynb``` for codes related to training the model, and plotting results, ```dataloader.py``` for data preprocessing steps, ```modelDCGAN.py``` for U-net classes and its architecture.




## Install Requirements:
For installing the requirements for this software, please run the following: 

 ```
 pip install -r requirements.txt
 ```
  
## Results:

Some of the results are shown below, comapring the ground truth masks to the predicted masks for same X-Ray images in the test dataset.

| X-ray Image  | Actual Mask  | Predicted Mask |
| ------------ | ------------ | ---------------|
| ![Alt text](image.png)| ![Alt text](image-1.png) |![Alt text](image-2.png)|


## Dataset
https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels/data