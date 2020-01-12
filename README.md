# Matrix-Capsule-Networks-PyTorch
![](https://miro.medium.com/max/1433/1*UVDimCXv0NxcucrFDgr1Xg.png)

A PyTorch implementation of Matrix capsule networks by Hinton et al.

I used the https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch code as a starter and improved the accuracy achieved by the code ,and also i made the training code more friendly.You can use your own dataset with this implementation, just replace the dataset and data_folder variables in the main notebook with the path to your own data. Additionally, I have uploaded the weight of the model in this repo you can find it under the name of model_model222_A_103.pth.

This implementation achieves **92%** accuracy on smallNORB dataset, which is the highest accuracy achieved by a pytorch implementation. Based on the Hinton's interview with the openreview, I used a weight decay of 2e-7 and exponential learning decay with a factor of 0.96.
This implemetation uses A=64, B=8, C=16, D=16, K=3 and 2 number of EM routing iterations for the nework. Also, the initial learning rate is set to 3e-3.Batch size of 32 has been used for the training. 

The training is done on a NVIDIA GTX 1070 TI. Each epoch takes about 16-17 mins.


