# -*- coding=utf-8 -*-
'''
Implementation of DMN/DMN-PRF for conversational response ranking in information-seeking conversation

Reference:
Liu Yang, Minghui Qiu, Chen Qu, Jiafeng Guo, Yongfeng Zhang, W. Bruce Croft, Jun Huang, Haiqing Chen.
Response Ranking with Deep Matching Networks and External Knowledge in Information-seeking Conversation Systems. SIGIR 2018.

@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
@homepage: https://sites.google.com/site/lyangwww/
'''
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam
from model import BasicModel

import sys
sys.path.append('../matchzoo/layers/')
sys.path.append('../matchzoo/utils/')
from Match import *
from utility import *


class DMN_CNN_MTL_IntentPrediction(BasicModel):
    def __init__(self, config):
        super(DMN_CNN_MTL_IntentPrediction, self).__init__(config)
        self.__name = 'DMN_CNN_MTL_IntentPrediction'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen', 'text1_max_utt_num',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        self.i = 0
        K.set_image_data_format('channels_last')

        if not self.check():
            raise TypeError('[DMN_CNN_MTL_IntentPrediction] parameter check wrong')
        print '[DMN_CNN_MTL_IntentPrediction] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_size', 32)
        self.set_default('topk', 100)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        def slice_reshape(x):
            print 'self.i', self.i, self.config['text1_maxlen']
            x1 = K.tf.slice(x, [0, self.i, 0], [-1, 1, self.config['text1_maxlen']])
            x2 = K.tf.reshape(tensor=x1, shape=(-1, self.config['text1_maxlen']))
            return x2

        def concate(x):
            return K.tf.concat([xx for xx in x], axis=3)

        def stack(x):
            return K.tf.stack([xx for xx in x], axis=1)

        query = Input(name='query', shape=(self.config['text1_max_utt_num'], self.config['text1_maxlen'],))
        # show_layer_info('Input query', query)
        # show_layer_info('Input doc', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)

        query_embeddings = []
        query_bigru_reps = []

        for i in range(self.config['text1_max_utt_num']):
            self.i = i
            query_cur_utt = Lambda(slice_reshape)(query)
            show_layer_info('query_cur_utt', query_cur_utt)

            # Transform current utterance in embedding
            q_embed = embedding(query_cur_utt)
            query_embeddings.append(q_embed)
            # show_layer_info('Query Embedding', q_embed)

            q_rep = Bidirectional(
                GRU(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))(q_embed)

            query_bigru_reps.append(q_rep)


        out_clf = Dense(self.config['max_intent'], activation='softmax')(K.tf.convert_to_tensor(query_bigru_reps[-1]))
        model_clf = Model(inputs=query, outputs=out_clf)

        return model_clf
