#!/usr/bin/env bash

for ((i=0;i<10;i++)); do
    cp data/mantis_10/crossval_intents/intent_data_train.tsv.$i data/mantis_10/ModelInput/intent_data_train.tsv && \
    cp data/mantis_10/crossval_intents/intent_data_dev.tsv.$i data/mantis_10/ModelInput/intent_data_dev.tsv && \
    cp data/mantis_10/crossval_intents/intent_data_test.tsv.$i data/mantis_10/ModelInput/intent_data_test.tsv && \
    cd matchzoo/conqa && \
    python2.7 preprocess_dmn_only_intents.py mantis_10 && \
    python2.7 gen_w2v_mikolov.py mantis_10 0 dmn_model_input && \
    python2.7 gen_w2v_filtered.py \
    ../../data/mantis_10/ModelInput/dmn_model_input/train_word2vec_mikolov_200d_no_readvocab.txt \
    ../../data/mantis_10/ModelInput/dmn_model_input/word_dict.txt \
    ../../data/mantis_10/ModelInput/cut_embed_mikolov_200d_no_readvocab.txt && \
    cd .. && \
    python2.7 main_conversation_qa.py --gpu_id 7 --phase train --model_file config/mantis_10/dmn_cnn_only_intents.config --or_cmd True && \
    python2.7 main_conversation_qa.py --gpu_id 7 --phase predict --model_file config/mantis_10/dmn_cnn_only_intents.config --or_cmd True && \
    cd .. && \
    mv data/mantis_10/ModelRes/dmn_cnn_only_intents.predict.test.txt data/mantis_10/ModelRes/dmn_cnn_only_intents.predict.test.txt.$i
done