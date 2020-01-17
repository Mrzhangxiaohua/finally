# DWMD
DimensionalWeighted Order-wise Moment Discrepancy for Domain-specific Hidden Representation Matching

In the source code, the DWMD domain-regularizer is denoted by 'DWMD'.

# Requirements
The implementation is based on Tensorflow and the neural networks library Keras. Required environment dependencies refer to [environment.yaml](environment.yaml) file

# Datasets
We report results for two different benchmark datasets in our paper: AmazonReview and Office. 

In addition:

* Office10 dataset set can be downloaded from https://drive.google.com/open?id=1NUV7BzjqVHQ0H1mp2o4zEV0PCIwGHJbq
* Office31 dataset set can be downloaded from https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view
* image_clef dataset set can be downloaded from https://drive.google.com/open?id=18U1OI5DkRBCKQICridUQvEuvnjZ-RRYL 

For office10 dataset,Copy the folders amazon, dslr and webcam to utils/office_dataset/. The network weights are taken from Keras's own weights. If you want to use your own weights, put the .h5 file in utils / office_dataset /.


# Experiments

We use unreliable .py files to run experiments on unreachable networks. For example, to test with office10 using resnet, you need to run the exp_office10_resnet.py file to achieve this. In our paper, we mainly use DWMD and DWMD1. The others are used for testing and comparison. Please note that our network randomly samples and searches the entire network, so it will take a long time and a small difference in results. 

