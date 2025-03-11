The SNNs are trained using SLTT. Except for changing the random seed to 2023, all other training settings are consistent with those in the SLTT open-source repository. The trained model parameters have been uploaded to the model folder, and you can use them directly. Please first download the official SLTT repository, then place WFWC_SLTT.py in the root directory of the repository. You can reproduce the experimental results using the following command.

python WFWC_SLTT.py -T 4 -dataset cifar10 -data_dir ./your_path_to_dataset -model spiking_resnet18 -c 3 -device cuda:0 -r 0.92
python WFWC_SLTT.py -T 4 -dataset cifar100 -data_dir ./your_path_to_dataset -model spiking_resnet18 -c 3 -device cuda:0 -r 0.6
python WFWC_SLTT.py -T 6 -dataset imagenet -data_dir ./your_path_to_dataset -model spiking_nfresnet34 -c 3 -device cuda:0 -r 0.24
