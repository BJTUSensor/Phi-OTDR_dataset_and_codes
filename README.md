# phi-OTDR-dataset-and-codes-

This dataset contains six types of events, including background noises, digging, knocking, shaking, watering and walking, in total of 15,612 samples. The data is divided into training set and test set with a ratio of 8:2, and the detailed number of events is displayed in the readme file of the dataset. The dataset also contains label files. Since GitHub has a data upload size limit, we have uploaded the data to Google Drive and Baidu Netdisk (link in the rawdata file). 

We also publicize codes for two common baseline models, which are the SVM (support vector machine, 1D method) and CNN (convolutional neural network,  2D approach) models. The files, das_data_svm.py, get_das_data.py, and feature_extraction.py are for the SVM Model, while das_data_cnn.py, models.py, amd mydataset.py are for the CNN.
An extra feature_visualization.py file is used to directly observe the event features' distinguishability.

You are **welcome to use our codes and dataset for non-commercial scientifc reseach proposes**, but please do mention the their origin (our paper and Github). For commercial applications, please contact us.

We have submitted a related paper to IEEE Sensors Journal[1].

[1]. Xiaomin Cao, Yunsheng Su, Zhiyan Jin, Kuanglu YU, An Open Dataset of Ф-OTDR Events with Two Classification Models as Baselines, submitted.

First Online Date: 22:00 Beijing Time, Jun. 2nd, 2022
