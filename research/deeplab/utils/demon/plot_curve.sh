train_log='train_depth'
train_log='upsampler/train_upsampler_l1'
train_log='log/train_upsampler_l2'
train_log=$1
loss_name=$2

echo ${train_log}
echo ${loss_name}

python ~/Paddle/python/paddle/utils/plotcurve.py \
${loss_name} \
-i ./output/${train_log}.log \
-o ./output/${train_log}.png \
-v v2
