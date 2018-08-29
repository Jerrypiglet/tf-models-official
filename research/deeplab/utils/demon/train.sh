learning_loss=nqmse
#learning_loss=l2
gpu_id=1
log=./output/train_${learning_loss}.log
init_model=./output/tf_model_stage1.tar.gz

/usr/bin/python train.py \
--learning_loss=$learning_loss \
--init_model=${init_model} 2>&1 | tee $log
