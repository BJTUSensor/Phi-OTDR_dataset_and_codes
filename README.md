# phi-OTDR-dataset-and-codes-

This dataset contains six types of events, including background noise, digging, knocking, shaking, watering and walking, with a total of 15,612 samples. 

Since GitHub has a data upload size limit, we upload the data to Google Drive and Baidu Netdisk, Visit the link in the rawdata file.
The data is divided into train set and test set with a ratio of 8:2, and the detailed number is displayed in the readme of the dataset. The dataset also contains label files.

We also publicize codes for two common baseline models, which are the SVM (support vector machine, 1D method) and CNN (convolutional neural network,  2D approach) models
das_data_svm.py,  get_das_data.py, and feature_extraction.py are about SVM codes. das_data_cnn.py, models.py, amd mydataset.py are about CNN codes.
feature_visualization.py is to directly observe the event distinguishability.
