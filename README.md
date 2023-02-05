# Phi-OTDR_dataset_and_codes

This dataset contains six types of Phi-OTDR events, including background noises, digging, knocking, shaking, watering and walking, in total of 15,612 samples. The data is divided into training set and test set with a ratio of 8:2, and the detailed number of events is displayed in the readme file of the dataset. The dataset also contains label files. Since GitHub has a data upload size limit, we have uploaded the data to Google Drive and Baidu Netdisk (link in the rawdata file). 

We also publicize codes for two common baseline models, which are the SVM (support vector machine, 1D method) and CNN (convolutional neural network,  2D approach) models. The files, das_data_svm.py, get_das_data.py, and feature_extraction.py are for the SVM Model, while das_data_cnn.py, models.py, amd mydataset.py are for the CNN.
An extra feature_visualization.py file is used to directly observe the event features' distinguishability.

You are **welcome to use our codes and dataset for non-commercial scientific reseach proposes**, but please do mention the their origin (our paper and Github). For commercial applications, please contact us.

See more details [1].

[1]. Cao, X., Su, Y., Jin, Z., & Yu, K. (2023). An open dataset of φ-OTDR events with two classification models as baselines. _Results in Optics_, 100372.

First Online Date: 22:00 Beijing Time, Jun. 2nd, 2022


----------update: Sept-13-2022--------------

This dataset contains six types of Phi-OTDR events, including background noises (3094 samples, Fig (a)), digging (2512 samples, Fig (b)), knocking (2530 samples, Fig (c)), shaking (2298, Fig (d)), watering (2728, Fig (e)) and walking (2450, Fig (f)), in a total of 15,612 samples. And the typical differentiated samples (size: 12(space)*9999(time)) are demonstrated in the figure.

![text](https://github.com/BJTUSensor/Phi-OTDR_dataset_and_codes/blob/main/figure.png?raw=true)
Fig. Time-space figure of typical samples of different events

To ensure the robustness of the dataset, two segments of fibers (5.1 km and 10.1 km) are used for collecting the above mentioned events at their tail parts (from 5.0 to 5.05 km and from 10.0 to 10.05 km) by ten members of our research team at different time. In order to facilitate subsequent data processing, we clip the collected data and only the points around the disturbance position (mostly at the center) are selected to make the samples. To be more specific, each sample of each event is composed of 10000 points in time domain (0.8 s for 5 km, and 1.25 s for 10 km), and 12 adjacent spatial points (10 m/point) in space domain.

The raw data is divided into training set and test set with a ratio of 8:2. And the label files for the samples are also provided within the dataset files. The format of the samples in the dataset is .mat file.
