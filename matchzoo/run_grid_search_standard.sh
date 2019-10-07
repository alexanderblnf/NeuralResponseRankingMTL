#!/bin/bash

NUM_ITERS=700
TASK=standard
GPU_ID=0
CONFIG_FILE=dmn_cnn.config
#========================================#
#              mantis_10                 #
#========================================#
for LEARNING_RATE in 0.001 0.0001 0.00001
do
  echo 'Learning Rate: '$LEARNING_RATE
  OUTPUT_FILE='run_grid_search_'$TASK'_rate_'$LEARNING_RATE'.log'
  for RANDOM_SEED in 10 100 1000 10000 100000
  do
      python2.7 main_conversation_qa.py --gpu_id ${GPU_ID} --phase train --model_file config/mantis_10/${CONFIG_FILE} --or_cmd True --num_iters ${NUM_ITERS} \
      --predict True --keras_random_seed ${RANDOM_SEED} --learning_rate ${LEARNING_RATE} >> $OUTPUT_FILE 2>&1 && sleep 5
  done
done
