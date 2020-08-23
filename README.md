# Deep-Learning-based-skin-cancer-classification

# 0.	Background and Data pre-processing
Early detection of skin lesions can help in its treatment and improve life. Most melanomas are ﬁrst identiﬁed visually, but unaided visual inspection only has a diagnostic accuracy of roughly 60%. Recently, medical professionals have used dermoscopy, a new technology of visual inspection, that both magniﬁes the skin and eliminates surface reﬂection.

These enhanced images help the dermatologist to diagnose melanoma more accurately. However, those human accuracy is about 65% to 75%.

With the rapid development of computer vision techniques, especially in the ﬁeld of deep learning, many Convolutional Neural Networks (CNNs) have been widely used in image classiﬁcation and detection. It is well acknowledged that CNNs show excellent performance in classifying images and achieve higher accuracy than many traditional methods.

Our proposed solution is to use the deep learning based models to help us in the skin cancer classification task and we would like to classify the most common skin cancers into 7 different classes.

More details could be found in https://challenge2018.isic-archive.com/ 

# 0.1 Installation and Setup

This code uses: Python 3.5, Keras 2.1.6, TensorFlwo 1.8.0. All other requirements are written in requirements.txt

Note: We use the development version of scikit-image for image resizing as it supports anti-aliasing. You can install development version directly from Github. Alternatively, you could change the resize function in load_image_by_id in datasets/ISIC2018/__init__.py to not use the anti-aliasing flag.

Note: Root directory can be edited in runs/paths.py: type in your own root directory path. Mine is '/OSM/CBR/D61_MLBSCD/results/David2'

# 0.2 Data preparation

Create the following subfolders in the folder ‘data’. Place the unzipped ISIC 2018 image data in the corresponding subfolders. 

•	ISIC2018_Task1-2_Training_Input
•	ISIC2018_Task1-2_Validation_Input
•	ISIC2018_Task1-2_Test_Input
•	ISIC2018_Task1_Training_GroundTruth
•	ISIC2018_Task3_Training_GroundTruth 
    o	Include ISIC2018_Task3_Training_LesionGroupings.csv and ISIC2018_Task3_Training_GroundTruth.csv file.
•	ISIC2018_Task3_Training_Input
•	ISIC2018_Task3_Validation_Input
•	ISIC2018_Task3_Test_Input

# 0.3 Data pre-processing

We resize all the images to 224x224x3 size and store them in numpy file ‘.npy’ for ease/speed of processing. 

	Run datasets/ISIC2018/preprocess_data.py to do the pre-processing, alternatively, it will be done the first time you call a function that needs the pre-processed data, for example the training process later. This can take a few hours to complete. 

	Each same dataset only needs to be done once for the pre-processing. The image data after the pre-processing will be stored in data/cache folder.

	If want to delete: delete the corresponding ‘.npy’ file in the data/cache

# 1.	Image Segmentation

To create a binary mask for each of the training images, in order to eliminate the background and unimportant features such as the human hairs and skin colour. This includes 3 main steps: training the model with the ISIC2018 dataset with the mask groundtruth, evaluating the model performance (optional) and predicting the mask for the each of the image in the new dataset for classification problem.

# a.Segmentation model training

Method:

This solution uses an encoder and a decoder in a U-NET type structure. The encoder can be one the pretrained models such as vgg16, denseNet169 (we are using this one)

	Go to the directory: “David” (your main directory)
	Run ‘module load tensorflow’ (make sure tensorflow module is installed)
	Run ‘python3 runs/seg_train.py’

Pre-processing for 2590 images takes around 20 minutes:



