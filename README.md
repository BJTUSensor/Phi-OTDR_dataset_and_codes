# Phi-OTDR_dataset_and_codes

This dataset contains six types of Phi-OTDR events, including background noises, digging, knocking, shaking, watering and walking, in total of 15,612 samples. The data is divided into training set and test set with a ratio of 8:2, and the detailed number of events is displayed in the readme file of the dataset. The dataset also contains label files. Since GitHub has a data upload size limit, we have uploaded the data to Google Drive and Baidu Netdisk (link in the rawdata file). 

We also publicize codes for two common baseline models, which are the SVM (support vector machine, 1D method) and CNN (convolutional neural network,  2D approach) models. The files, das_data_svm.py, get_das_data.py, and feature_extraction.py are for the SVM Model, while das_data_cnn.py, models.py, amd mydataset.py are for the CNN.
An extra feature_visualization.py file is used to directly observe the event features' distinguishability.

You are **welcome to use our codes and dataset for non-commercial scientific reseach proposes**, but please do mention the their origin (our paper and Github). For commercial applications, please contact us.

We have submitted a related paper to Elsevier's Optics & Laser Technology[1].

[1]. Xiaomin Cao, Yunsheng Su, Zhiyan Jin, Kuanglu YU, An Open Dataset of Ф-OTDR Events with Two Classification Models as Baselines, submitted.

First Online Date: 22:00 Beijing Time, Jun. 2nd, 2022


------update: Sept-10-2022
This dataset contains six types of Phi-OTDR events, including background noises (3094 samples, Fig (a)), digging (2512 samples, Fig (b)), knocking (2530 samples, Fig (c)), shaking (2298, Fig (d)), watering (2728, Fig (e)) and walking (2450, Fig (f)), in a total of 15,612 samples. And the typical differentiated samples (size: 12(space)*9999(time)) are demonstrated in the figure.

