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

•	Total memory required: 16.9GB
•	Total params:17,937,121
•	Total trainable params: 17,937,121
•	Total non-trainable params: 0
•	Total training time: 210 min for 5 folds
•	Input shape: (224,224,1)
•	Number of training sample: 2072
•	Number of validation sample: 518
•	Pretrained model: DenseNet169 (can be used another models, or ensembles see ISIC2018 leader board)
•	Loss: binary_crossentropy
•	Encoder activation: Relu
•	Decoder activation: Relu
•	Batch size: 32
•	Epochs : 35
•	Initial learning rate: 0.001
•	Minimum learning rate: 1e-07
•	Use data augument = True
•	Use multi-processing = True
•	Batch normalization = True
•	The training performance of the model after each epoch will be saved in ‘model_data’ directory as a .csv file. The weights will be saved in the same directory too.
•	Model weight will be saved to /OSM/CBR/D61_MLBSCD/results/David2/model_data/task1_densenet169_k0_v0/task1_densenet169_k0_v0.hdf5
Where denseNet169 is the model backbone’s name, k0 is the 1st folder, v0 is the 1st
version

Note: We are using the k-fold cross validation method. See: https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833 for more detail. Set the variable: num_folds in ‘seg_train.py’ to 5 if you want to do 5 fold training. Set it to 1 if you want to use a single fold.

Note: after pre-processing, images data will be saved in data/cache as ‘.npy’

Note: Effect of Batch Normalization: https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd

# 1.2 Segmentation Evaluation:

	Run ‘python3 runs/seg_eval.py’

	Results will be shown in the terminal

All the settings for the segmentation model and the classification model are in the runs/models/__init__.py, if you would like to change

Change the code ‘k_fold = ’, ‘version = ’ in runs/seg_eval.py to evaluate a specific folder

•	Eval on the 1st fold: mean jaccard: 0.796, threshold jaccard: 0.748
•	Eval on the 2nd fold: mean jaccard: 0.789, threshold jaccard: 0.743
•	Eval on the 3rd fold: mean jaccard: 0.794, threshold jaccard: 0.744
•	Eval on the 4th fold: mean jaccard: 0.798, threshold jaccard: 0.749
•	Eval on the 5th fold: mean jaccard: 0.801, threshold jaccard: 0.749



# 1.3	 Segmentation Prediction (pre-request before classification training)

Make a segmentation mask for each of the image, make the mask, crop the mask with a bounding box function, then record the coordinate and crop the bounding box in the original image as well, which result in two images with same size. 

Then multiply the cropped mask with the cropped original image. The main reason for doing this it to reduce any unimportant features such as human hair around the moles before feeding into the mole classification deep learning model. This could significantly improve the accuracy.

	Run ‘python3 runs/seg_predict.py’ to output the segmented and cropped images from the original images in the ‘submission’ folder

	Alternatively, run ‘python3 runs/seg_predict.py mask’ if you only want the pure mask to be saved in the ‘submission’ folder

	Results will be shown in the submissions folder

 
•	Predict on ISIC2018_Task3_Training_Input to generate the segmented and cropped images for each training input for classification into folder submissions/task1_test
•	Total image: 3373
•	Total processing time: 6 min
•	Backbone name: Vgg16
•	Use TTA=False if not want to use test time augmentation (which uses rotations of the image and averages the predictions)
•	Set the variable: num_folds in ‘seg_train.py’ to 5 if you want to do 5 fold training. Set it to 1 if you want to use a single fold.
•	Results will appear in the submission folder, with segmented and cropped images 
•	In the seg_predict.py, choose ‘validation’ or ‘test’ in ‘pred_set = ‘ to make prediction on the validation set or the test set.
Note:  To visualize the mask or save the mask to a new folder, you have to change few settings, otherwise the cropped and segmented images will be saved instead rather than the pure masks.

# 2	Image Classification

# 2.1 Image Classification model train:
	Run ‘python3 runs/cls_train.py’

•	Dropout rate: 0
•	Learning rate: 1e-4
•	Number of folds: 5
•	Pretrained model: DenseNet201
•	Training input: 8023 pictures
•	Validation input: 2001 pictures
•	Batch size: 16
•	Epochs: 30
•	Because the data is unbalanced, use 2 methods: 1. Focal-loss 2. Use unequal sample weights (balanced weight)
•	InceptionV3: the accuracy is about 78%
•	DenseNet201: the accuracy is about 83% (batch size:16)
•	Dense layer regularizer = L2
•	Class_with_type = ‘balanced’ (can be chosen from ‘ones’, ‘balanced’, ‘ balanced-sqrt)
•	Compare the validation loss for each epoch, if it is less then save the weights, in the prediction process it will retrieve the weights with the minimum loss.
•	Set the variable: num_folds in ‘cls_train.py’ to 5 if you want to do 5 fold training. Set it to 1 if you want to use a single fold.
•	The training performance of the model after each epoch will be saved in ‘model_data’ directory as a .csv file. The weights will be saved in the same directory too.


Note: As the training data set has imbalanced samples among classes. We have used some techniques to handle this imbalanced dataset, such as balanced weight, focal loss and dropout. See: https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758

Note: Task 3 refers to the image classification process, Task 1 refers to the image segmentation process

# 2.2 Classification model evaluation: 

	Run ‘python3 runs/cls_eval.py’

	Results will be shown in the terminal

# 2.3 Classification model prediction:

	Run ‘python3 runs/cls_predict.py’

	Results will be shown in the submissions folder as a ‘.csv’ file

•	Use TTA=False if not want to use test time augmentation (which uses rotations of the image and averages the predictions)
•	Set the variable: num_folds in ‘cls_train.py’ to 5 if you want to do 5 fold training. Set it to 1 if you want to use a single fold.
•	Results will appear in the submission folder, it is a .csv file with image ids in the first column and the probability for being each class in the following columns
•	In the cls_predict.py, choose ‘validation’ or ‘test’ in ‘pred_set = ‘ to make prediction on the validation set or the test set.

# 3	Future work

1.	It is recommended to use model ensembles to achieve a better result in the prediction
2.	The edge of the cropped and segmented image can be smoothening by trying other algorithms 
3.	Try to save the image into other image formats to improve the overall picture quality
4.	Some more state-of-the-art models are still being released in 2019 and 2020
5.	The results pages can be made into other ways rather than a single csv file

