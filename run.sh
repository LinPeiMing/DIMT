CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4329 --use-env\
basicsr/train.py -opt options/train/DIMT.yml --launcher pytorch

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4329 --use-env\
#  basicsr/train.py -opt options/train/DIMT-GAN.yml --launcher pytorch
