# Deep-Learning-based-skin-cancer-classification
0.	Data pre-processing

0.1 Installation and Setup:
This code uses: Python 3.5, Keras 2.1.6, TensorFlwo 1.8.0. All requirements are written in requirements.txt
Note: We use the development version of scikit-image for image resizing as it supports anti-aliasing. You can install development version directly from Github. Alternatively, you could change the resize function in load_image_by_id in datasets/ISIC2018/__init__.py to not use the anti-aliasing flag.
Note: Root directory can be edited in runs/paths.py: type in your own root directory path. Mine is '/OSM/CBR/D61_MLBSCD/results/David2'

0.2 Data preparation:
Create the following subfolders in the folder ‘data’. Place the unzipped ISIC 2018 image data in the corresponding subfolders. 
•	ISIC2018_Task1-2_Training_Input
•	ISIC2018_Task1-2_Validation_Input
•	ISIC2018_Task1-2_Test_Input
•	ISIC2018_Task1_Training_GroundTruth
•	ISIC2018_Task3_Training_GroundTruth 
o	Include ISIC2018_Task3_Training_LesionGroupings.csv file. 
o	Include ISIC2018_Task3_Training_GroundTruth.csv file.
•	ISIC2018_Task3_Training_Input
•	ISIC2018_Task3_Validation_Input
•	ISIC2018_Task3_Test_Input

0.3 Data pre-processing:
We resize all the images to 224x224x3 size and store them in numpy file ‘.npy’ for ease/speed of processing. 
	Run datasets/ISIC2018/preprocess_data.py to do the pre-processing, or it will be done the first time you call a function that needs the pre-processed data. This can take a few hours to complete. 
	This only need to be done once. The image data after the pre-processing will be stored in data/cache folder.
	If want to delete: delete the corresponding ‘.npy’ file in the data/cache
1.	Image Segmentation

1.1 Segmentation model training: 
	Go to the directory: “David” 
	Run ‘module load tensorflow’ 
	Run ‘python3 runs/seg_train.py’
Pre-processing stage takes around 20 minutes 
 
•	Total memory required: 16.9GB
•	Total params:17,937,121
•	Total trainable params: 17,937,121
•	Total non-trainable params: 0
•	Total training time: 86 min for 5 folds
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
•	Use data augument: true
•	Use multi-processing: true
•	The training performance of the model after each epoch will be saved in ‘model_data’ directory as a .csv file. The weights will be saved in the same directory too.
•	Model weight will be saved to /OSM/CBR/D61_MLBSCD/results/David2/model_data/task1_densenet169_k0_v0/task1_densenet169_k0_v0.hdf5
Where denseNet169 is the model backbone’s name, k0 is the 1st folder, v0 is the 1st
version
 
Note: We are using the k-fold cross validation method. See: https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833. Set the variable: num_folds in ‘seg_train.py’ to 5 if you want to do 5 fold training. Set it to 1 if you want to use a single fold.
Note: after pre-processing images are in data/cache as ‘.npy’

1.2 Segmentation Evaluation:
	Run ‘python3 runs/seg_eval.py’
	Results will be shown in the terminal

•	Eval on the 1st fold: mean jaccard: 0.796, threshold jaccard: 0.748
•	Eval on the 2nd fold: mean jaccard: 0.789, threshold jaccard: 0.743
•	Eval on the 3rd fold: mean jaccard: 0.794, threshold jaccard: 0.744
•	Eval on the 4th fold: mean jaccard: 0.798, threshold jaccard: 0.749
•	Eval on the 5th fold: mean jaccard: 0.801, threshold jaccard: 0.749












1.3 Segmentation Prediction (pre-request before classification training)
	Run ‘python3 runs/seg_predict.py’
	Results will be shown in the submissions folder


•	Predict on ISIC2018_Task3_Training_Input to generate the segmented and cropped images for each training input for classification into folder submissions/task1_test
•	In runs/datasets/ISIC2018/__init__.py line 143 ‘io.imread()’ changes to ‘cv2.imread()’ 
•	image = io.imread(img_fname) #suitable for seg_predict / seg_train to read an original image
•	image = cv2.imread(img_fname) # suitable for cls_train / cls_predict to read a segmented and cropped image
•	Total image: 3373
•	Total processing time: 6 min
•	Backbone name: Vgg16
•	Use TTA=False if not want to use test time augmentation (which uses rotations of the image and averages the predictions)
•	Set the variable: num_folds in ‘seg_train.py’ to 5 if you want to do 5 fold training. Set it to 1 if you want to use a single fold.
•	Results will appear in the submission folder, with segmented and cropped images 
•	In the seg_predict.py, choose ‘validation’ or ‘test’ in ‘pred_set = ‘ to make prediction on the validation set or the test set.
Note: Reason for image segmentation before the classification work: Make a segmentation for each of the image, make the mask, crop the mask with a bounding box function, then record the coordinate and crop the bounding box in the original image as well, which result in two images with same size. Then multiply the cropped mask with the cropped original image. The main reason for doing this it to reduce any unimportant features such as human hair around the moles before feeding into the mole classification deep learning model. This could significantly improve the accuracy.





 



 


 





2.	Image Classification

2.1 Image Classification model train:
	Run ‘python3 runs/cls_train.py’

•	Dropout rate: 0
•	Learning rate: 1e-4
•	Number of folds: 5
•	Pretrained model: DenseNet201
•	Training input: 8023 pictures
•	Validation input: 2001 pictures
•	Batch size: 16
•	Epochs: 50
•	Errors: all input array must have the same shape
•	Because the data is unbalanced, use 2 methods: 1. Focal-loss 2. Use unequal sample weights (balanced weight)
•	InceptionV3: the accuracy is about 78%
•	DenseNet201: the accuracy is about 83% (batch size:16)
•	Compare the validation loss for each epoch, if it is less then save the weights, in the prediction process it will retrieve the weights with the minimum loss.
•	Set the variable: num_folds in ‘cls_train.py’ to 5 if you want to do 5 fold training. Set it to 1 if you want to use a single fold.
•	The training performance of the model after each epoch will be saved in ‘model_data’ directory as a .csv file. The weights will be saved in the same directory too.


2.2 Classification model evaluation:
	Run ‘python3 runs/cls_eval.py’
	Results will be shown in the terminal



2.3 Classification model prediction:
	Run ‘python3 runs/cls_predict.py’
	Results will be shown in the submissions folder

•	Use TTA=False if not want to use test time augmentation (which uses rotations of the image and averages the predictions)
•	Set the variable: num_folds in ‘cls_train.py’ to 5 if you want to do 5 fold training. Set it to 1 if you want to use a single fold.
•	Results will appear in the submission folder, it is a .csv file with image ids in the first column and the probability for being each class in the following columns
•	In the cls_predict.py, choose ‘validation’ or ‘test’ in ‘pred_set = ‘ to make prediction on the validation set or the test set.


