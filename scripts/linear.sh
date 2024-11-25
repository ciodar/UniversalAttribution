device=cuda:0
backbone=vit_base_patch16_clip_224.laion2b
dataset=GenImage
config_name=base

for id in 1 2 3 4 5
do
python3 linear.py \
--config_name $config_name \
--backbone $backbone \
--device $device \
--data ${dataset}:split=split${id}
done