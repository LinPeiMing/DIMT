CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4323 --use-env \
basicsr/test.py -opt options/test/swinFIRSSR.yml --launcher pytorch

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4327 --use-env \
#basicsr/test.py -opt options/test/NAFSSR.yml --launcher pytorch

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4323 --use-env \
#basicsr/test.py -opt options/test/test_SCGLANet-L_4x_Track3.yml --launcher pytorch
