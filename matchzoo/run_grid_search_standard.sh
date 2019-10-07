#!/bin/bash

NUM_ITERS=700
#========================================#
#              mantis_10                 #
#========================================#
for LEARNING_RATE in 0.001 0.0001 0.00001
do
  echo 'Learning Rate: '$LEARNING_RATE
  for RANDOM_SEED in 10 100 1000 10000 100000
  do
      python2.7 main_conversation_qa.py --gpu_id 0 --phase train --model_file config/mantis_10/dmn_cnn.config --or_cmd True --num_iters ${NUM_ITERS} \
      --predict True --keras_random_seed ${RANDOM_SEED} --learning_rate ${LEARNING_RATE}&& sleep 5
  done
done
