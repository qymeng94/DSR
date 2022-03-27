# DSR
This repository is the official PyTorch implementation of paper: Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation (CVPR 2022).
## Dependencies
- Python >= 3.6
- PyTorch >= 1.7.1
- torchvision
- Python packages: `pip install numpy matplotlib progress`


## Training
Please run the following code. The hyperparameters in the code are the same as in the paper.
###CIFAR-10

	python -u cifar/main_single_gpu.py --path ./data --dataset cifar10 --model preresnet.resnet18_lif --name [checkpoint_name]

	python -u cifar/main_single_gpu.py --path ./data --dataset cifar10 --model preresnet.resnet18_if --Vth 6 --alpha 0.5 --Vth_bound 0.01 --name [checkpoint_name]

###CIFAR-100

	python -u cifar/main_single_gpu.py --path ./data --dataset cifar100 --model preresnet.resnet18_lif --name [checkpoint_name]

	python -u cifar/main_single_gpu.py --path ./data --dataset cifar100 --model preresnet.resnet18_if --Vth 6 --alpha 0.5 --Vth_bound 0.01 --name [checkpoint_name]

###DVS-CIFAR10

	python -u cifar/main_single_gpu.py --path ./data/CIFAR10DVS --dataset CIFAR10DVS --model vgg.vgg11_lif --lr=0.05 --epochs=300 --name [checkpoint_name]

	python -u cifar/main_single_gpu.py --path ./data/CIFAR10DVS --dataset CIFAR10DVS --model vgg.vgg11_if --Vth 6 --alpha=0.5 --Vth_bound 0.01 --lr=0.05 --epochs=300 --name [checkpoint_name]

###Training with Multiple GPUs
For the CIFAR-10, CIFAR-100, and DVS-CIFAR10 tasks, multiple GPUs can also be used. The example code is shown below.

	python -u -m torch.distributed.launch --nproc_per_node [number_of_gpus] cifar/main_multiple_gpus.py --path ./data --dataset cifar10 --model preresnet.resnet18_lif --name [checkpoint_name]

###ImageNet
For the ImageNet classification task, we conduct [hybrid training](https://openreview.net/pdf?id=B1xSperKvH).

First, we train an ANN.

	python imagenet/main.py --arch preresnet_ann.resnet18 --data ./data/imagenet --name model_ann --optimizer SGD --wd 1e-4  --batch-size 256 --lr 0.1

Then, we calculate the maximum post-activation as the initialization for spike thresholds.

	python imagenet/main.py --arch preresnet_cal_Vth.resnet18  --data ./data/imagenet --pre_train model_ann.pth --calculate_Vth resnet18_Vth

Next, we train the SNN.

    python imagenet/main.py --dist-url tcp://127.0.0.1:20500 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --arch preresnet_snn.resnet18_if  --data ./data/imagenet --pre_train model_ann.pth --load_Vth resnet18_Vth.dict

The pretrained ANN model and calculated thresholds can be downloaded from [here](https://cuhko365-my.sharepoint.com/:u:/g/personal/219019044_link_cuhk_edu_cn/EYL7oYEXaO5Lmwyd1aMofG4BCzO-OndBZZFvtx4tdIsmTQ?e=7rKO25) and [here](https://cuhko365-my.sharepoint.com/:u:/g/personal/219019044_link_cuhk_edu_cn/EbCSmhOql4hNpHU3JFtIVBQB4gUk_V_wZDsFP-VVlIwH0A?e=Qsuc1A). Please put them in the path ./checkpoint/imagenet.
## Credits

The code for the data preprocessing of DVS-CIFAR10 is based on the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repo. The code for some utils are from the [pytorch-classification](https://github.com/bearpaw/pytorch-classification) repo.


