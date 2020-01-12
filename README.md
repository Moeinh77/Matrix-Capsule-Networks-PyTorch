# Matrix-Capsule-Networks-PyTorch
A PyTorch implementation of Matrix capsule networks by Hinton et al.

This implementation achieves **92%** accuracy on smallNORB dataset, which is the highest accuracy achieved by a pytorch implementation. Based on the Hinton's interview with the openreview, I used a weight decay of 2e-7 and exponential learning decay with a factor of 0.96. Also, the initial learning rate is set to 3e-3.

Batch size of 32 has been used for the training. Training is done on a NVIDIA GTX 1070 TI. Each epoch takes about 16-17 mins.
