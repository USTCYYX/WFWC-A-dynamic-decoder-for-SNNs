The pre-trained model is provided by the QCFS open-source repository, and you can download the model from the following link:

https://drive.google.com/drive/folders/1P-2egAraWtsQYNzp8lcJvZVEG_KLVV5Q?usp=sharing

Please download the SNN_conversion_QCFS repository, then place the WFWC_QCFS.py file in the root directory of that repository. Place the model you downloaded in the model folder. Run the following command to reproduce the results from the paper.

python WFWC_QCFS.py --model=vgg16 --data=cifar10 --id=cifar10-vgg16-example --t=30 --device=cuda:2 --l=16 --c=5 --r=0.99991

python WFWC_QCFS.py --model=vgg16 --data=cifar100 --id=cifar100-vgg16-l8-example --t=30 --device=cuda:2 --l=8 --c=6 --r=0.51


