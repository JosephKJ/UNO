# CIFAR10 (5 labeled classes, 5 unlabeled classes)
#python main_discover.py --dataset CIFAR10 --gpus 1 --max_epochs 200 --batch_size 512 --num_labeled_classes 5 --num_unlabeled_classes 5 --pretrained ./models/pretrain-resnet18-CIFAR10.cp --num_heads 4 --comment 5_5 --precision 16 --multicrop --overcluster_factor 10 --download

# CIFAR100-20 (80 labeled classes, 20 unlabeled classes)
#python main_discover.py --dataset CIFAR100 --gpus 1 --max_epochs 200 --batch_size 512 --num_labeled_classes 80 --num_unlabeled_classes 20 --pretrained ./models/pretrain-resnet18-CIFAR100-80_20.cp --num_heads 4 --comment 80_20 --precision 16 --multicrop --overcluster_factor 5 --download

# CIFAR100-50 (50 labeled classes, 50 unlabeled classes)
#python main_discover.py --dataset CIFAR100 --gpus 1 --max_epochs 500 --batch_size 512 --num_labeled_classes 50 --num_unlabeled_classes 50 --pretrained ./models/pretrain-resnet18-CIFAR100-50_50.cp --num_heads 4 --comment 50_50 --precision 16 --multicrop

# ImageNet (882 labeled classes, 30 unlabeled classes)
python main_discover.py --dataset ImageNet --gpus 2 --distributed_backend ddp --sync_batchnorm --precision 16 --data_dir /home/josephkj/imagenet/ --max_epochs 60 --base_lr 0.2 --warmup_epochs 5 --batch_size 256 --num_labeled_classes 882 --num_unlabeled_classes 30 --num_heads 4 --pretrained ./models/pretrain-resnet18-ImageNet.cp --imagenet_split A --comment 882_30-A --overcluster_factor 4 --multicrop
