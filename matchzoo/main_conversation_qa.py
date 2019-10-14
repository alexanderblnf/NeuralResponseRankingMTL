# -*- coding: utf8 -*-
import os

seed = None

import sys
import time
import json
import argparse

import random
if seed:
    random.seed(seed)

import numpy
if seed:
    numpy.random.seed(seed)

import tensorflow
if seed:
    tensorflow.set_random_seed(seed)

from collections import OrderedDict
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.optimizers import Adam
from utils import *
import inputs
import metrics
from losses import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import keras.backend.tensorflow_backend as ktf
#
#
# def get_session(gpu_fraction=0.8):
#     gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
#     return tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=gpu_options))
#
#
# ktf.set_session(get_session())


def custom_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)

def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']['setting']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, config['model']['model_path'])

        model = import_object(config['model']['model_py'], model_config)
        mo = model.build()

    return mo


def train(config):

    if seed is None:
        raise Exception('Seed should be set')
    print('Using seed: ' + str(seed))
    # read basic config
    global_conf = config["global"]
    learning_rate = global_conf['learning_rate']
    use_existing_weights = global_conf['use_existing_weights'] if 'use_existing_weights' in global_conf else None
    optimizer = Adam(lr=learning_rate)
    weights_file = str(global_conf['weights_file']) + '.%d'
    weights_file_web = str(global_conf['weights_file_web']) + '.%d' if 'weights_file_web' in global_conf else None
    display_interval = int(global_conf['display_interval'])
    num_iters = int(global_conf['num_iters'])
    save_weights_iters = int(global_conf['save_weights_iters'])

    # read input config
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print '[Embedding] Embedding Load Done.'

    # list all input tags and construct tags config
    input_train_conf = OrderedDict()
    input_eval_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue

        if input_conf[tag]['phase'] == 'TRAIN':
            input_train_conf[tag] = {}
            input_train_conf[tag].update(share_input_conf)
            input_train_conf[tag].update(input_conf[tag])
        elif input_conf[tag]['phase'] == 'EVAL':
            input_eval_conf[tag] = {}
            input_eval_conf[tag].update(share_input_conf)
            input_eval_conf[tag].update(input_conf[tag])

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag != 'share' and input_conf[tag]['phase'] == 'PREDICT':
            continue
        if 'text1_corpus' in input_conf[tag]:
            datapath = input_conf[tag]['text1_corpus']
            if datapath not in dataset:
                dataset[datapath] = read_data_2d(datapath)
        if 'text2_corpus' in input_conf[tag]:
            datapath = input_conf[tag]['text2_corpus']
            if datapath not in dataset:
                dataset[datapath] = read_data_2d(datapath)
        if 'qa_comat_file' in input_conf[tag]: # qa_comat_file for qa_cooccur_matrix in DMN_KD
            datapath = input_conf[tag]['qa_comat_file']
            if datapath not in dataset:
                dataset[datapath] = read_qa_comat(datapath)

    print '[Dataset] %s Dataset Load Done.' % len(dataset)

    # initial data generator
    train_gen = OrderedDict()
    eval_gen = OrderedDict()

    for tag, conf in input_train_conf.items():
        print conf
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        if 'qa_comat_file' in share_input_conf:
            conf['qa_comat'] = dataset[conf['qa_comat_file']]

        generator = inputs.get(conf['input_type'])
        train_gen[tag] = generator(config=conf)

    for tag, conf in input_eval_conf.items():
        print conf
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]

        if 'qa_comat_file' in share_input_conf:
            conf['qa_comat'] = dataset[conf['qa_comat_file']]

        generator = inputs.get(conf['input_type'])
        eval_gen[tag] = generator(config=conf)

    ######### Load Model #########

    if config['net_name'] == 'DMN_CNN_MTL':
        model, model_clf = load_model(config)
        if use_existing_weights:
            weights_file_to_load = str(global_conf['weights_file']) + '.' + str(
                global_conf['weights_to_load']) + '-' + str(seed)
            model.load_weights(weights_file_to_load)

        model_clf.compile(optimizer=optimizer, loss=custom_loss)
        print '[Model] MTL models Compile Done.'
    elif config['net_name'] == 'DMN_CNN_INTENTS':
        model_clf = load_model(config)
        model_clf.compile(optimizer=optimizer, loss=custom_loss)
        print '[Model] Intent Only classifier model Compile Done.'
    elif config['net_name'] == 'DMN_CNN_MTL_Web' or config['net_name'] == 'DMN_CNN_MTL_Web_v2':
        model, model_web = load_model(config)
        if use_existing_weights:
            weights_file_to_load = str(global_conf['weights_file']) + '.' + str(
                global_conf['weights_to_load']) + '-' + str(seed)
            model.load_weights(weights_file_to_load)
            weights_file_web = str(global_conf['weights_file_web']) + '.' + str(
                global_conf['weights_to_load']) + '-' + str(seed)
            model_web.load_weights(weights_file_web)
    elif config['net_name'] == 'DMN_CNN_MTL_All':
        model, model_web, model_clf = load_model(config)
        model_clf.compile(optimizer=optimizer, loss=custom_loss)
    else:
        model = load_model(config)
        if use_existing_weights:
            weights_file_to_load = str(global_conf['weights_file']) + '.' + str(
                global_conf['test_weights_iters']) + '-' + str(seed)
            model.load_weights(weights_file_to_load)

        print '[Model] Response Ranking model Compile Done.'

    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)

    if config['net_name'] != 'DMN_CNN_INTENTS':
        loss = []
        for lobj in config['losses']:
            if lobj['object_name'] in mz_specialized_losses:
                loss.append(rank_losses.get(lobj['object_name'])(lobj['object_params']))
            else:
                loss.append(rank_losses.get(lobj['object_name']))

        model.compile(optimizer=optimizer, loss=loss)
        print '[Model] Model Compile Done.'

        if config['net_name'] == 'DMN_CNN_MTL_Web' or config['net_name'] == 'DMN_CNN_MTL_Web_v2' \
                or config['net_name'] == 'DMN_CNN_MTL_All':
            model_web.compile(optimizer=optimizer, loss=loss)
            print('[Model Web] Model Compile Done')

    if share_input_conf['predict'] == 'False':
        if 'test' in eval_gen:
            del eval_gen['test']
        if 'valid' in eval_gen:
            del eval_gen['valid']
        if 'eval_predict_in' in eval_gen:
            del eval_gen['eval_predict_in']

    for i_e in range(num_iters):
        for tag, generator in train_gen.items():
            genfun = generator.get_batch_generator()
            print '[%s]\t[Train:%s]' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag),

            if tag == "train_clf":
                correct_model = model_clf
            elif tag == 'train_web':
                correct_model = model_web
            elif tag == "train":
                correct_model = model

            history = correct_model.fit_generator(
                genfun,
                steps_per_epoch=display_interval,  # if display_interval = 10, then there are 10 batches in 1 epoch
                epochs=1,
                shuffle=False,
                verbose=0)

            print 'Iter:%d\tloss=%.6f' % (i_e, history.history['loss'][0])

        if (i_e+1) % save_weights_iters == 0:
            for tag, generator in eval_gen.items():
                print('Evaluating tag:' + str(tag))
                genfun = generator.get_batch_generator()
                print '[%s]\t[Eval:%s]' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag),
                res = dict([[k,0.] for k in eval_metrics.keys()])
                num_valid = 0

                if tag == "valid":
                    correct_model = model
                elif tag == "valid_web":
                    correct_model = model_web
                elif tag == "valid_clf":
                    correct_model = model_clf

                for input_data, y_true in genfun:
                    y_pred = correct_model.predict(input_data, batch_size=len(y_true))
                    if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                        list_counts = input_data['list_counts']
                        for k, eval_func in eval_metrics.items():
                            for lc_idx in range(len(list_counts)-1):
                                pre = list_counts[lc_idx]
                                suf = list_counts[lc_idx+1]
                                res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])
                        num_valid += len(list_counts) - 1
                    else:
                        for k, eval_func in eval_metrics.items():
                            res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                        num_valid += 1
                generator.reset()
                print 'Iter:%d\t%s' % (i_e, '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()]))
                sys.stdout.flush()

        sys.stdout.flush()

        weights_file_name = (weights_file % (i_e+1)) + '-' + str(seed)
        if (i_e+1) % save_weights_iters == 0:
            if config['net_name'] == 'DMN_CNN_MTL_Web' or config['net_name'] == 'DMN_CNN_MTL_Web_v2'  \
                    or config['net_name'] == 'DMN_CNN_MTL_All':
                weights_file_name_web = (weights_file_web % (i_e + 1)) + '-' + str(seed)
                model.save_weights(weights_file_name)
                model_web.save_weights(weights_file_name_web)
            elif config['net_name'] != 'DMN_CNN_INTENTS':
                model.save_weights(weights_file_name)
            else:
                model_clf.save_weights(weights_file_name)

def predict(config):
    ######## Read input config ########

    print(json.dumps(config, indent=2))
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print '[Embedding] Embedding Load Done.'

    # list all input tags and construct tags config
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print '[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys())

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag == 'share' or input_conf[tag]['phase'] == 'PREDICT':
            if 'text1_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text1_corpus']
                if datapath not in dataset:
                    dataset[datapath] = read_data_2d(datapath)
            if 'text2_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text2_corpus']
                if datapath not in dataset:
                    dataset[datapath] = read_data_2d(datapath)
            if 'qa_comat_file' in input_conf[tag]:  # qa_comat_file for qa_cooccur_matrix in DMN_KD_CQA and DMN_KD_Web
                datapath = input_conf[tag]['qa_comat_file']
                if datapath not in dataset:
                    dataset[datapath] = read_qa_comat(datapath)
    print '[Dataset] %s Dataset Load Done.' % len(dataset)

    # initial data generator
    predict_gen = OrderedDict()

    for tag, conf in input_predict_conf.items():
        print conf
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        if 'qa_comat_file' in share_input_conf:
            conf['qa_comat'] = dataset[conf['qa_comat_file']]
        generator = inputs.get(conf['input_type'])
        predict_gen[tag] = generator(
                                    #data1 = dataset[conf['text1_corpus']],
                                    #data2 = dataset[conf['text2_corpus']],
                                     config = conf )

    ######## Read output config ########
    output_conf = config['outputs']

    ######## Load Model ########
    global_conf = config["global"]
    weights_file = str(global_conf['weights_file']) + '.' + str(global_conf['test_weights_iters']) + '-' + str(seed)

    if config['net_name'] == 'DMN_CNN_MTL':
        model, model_clf = load_model(config)
        model.load_weights(weights_file)
    elif config['net_name'] == 'DMN_CNN_INTENTS':
        model_clf = load_model(config)
        model_clf.load_weights(weights_file)
    elif config['net_name'] == 'DMN_CNN_MTL_Web' or config['net_name'] == 'DMN_CNN_MTL_Web_v2':
        model, model_web = load_model(config)
        model.load_weights(weights_file)
        weights_file_web = str(global_conf['weights_file_web']) + '.' + str(global_conf['test_weights_iters']) + '-' + str(seed)
        model_web.load_weights(weights_file_web)
    elif config['net_name'] == 'DMN_CNN_MTL_All':
        model, model_web, model_clf = load_model(config)
        model.load_weights(weights_file)
        weights_file_web = str(global_conf['weights_file_web']) + '.' + str(
            global_conf['test_weights_iters']) + '-' + str(seed)
        model_web.load_weights(weights_file_web)
    else:
        model = load_model(config)
        model.load_weights(weights_file)

    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)

    res = dict([[k,0.] for k in eval_metrics.keys()])

    for tag, generator in predict_gen.items():
        genfun = generator.get_batch_generator()
        print '[%s]\t[Predict] @ %s ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag),
        num_valid = 0
        res_scores = {}

        if tag == 'predict':
            model_to_evaluate = model
        elif tag == 'predict_clf':
            model_to_evaluate = model_clf
        elif tag == 'predict_web':
            model_to_evaluate = model_web

        for input_data, y_true in genfun:
            y_pred = model_to_evaluate.predict(input_data, batch_size=len(y_true))
            
            if tag == 'predict_clf':
                y_pred = np.argmax(y_pred, axis=1)

            if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                list_counts = input_data['list_counts']
                for k, eval_func in eval_metrics.items():
                    for lc_idx in range(len(list_counts)-1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx+1]
                        res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])

                y_pred = np.squeeze(y_pred)
                for lc_idx in range(len(list_counts) - 1):
                    pre = list_counts[lc_idx]
                    suf = list_counts[lc_idx + 1]
                    for p, y, t in zip(input_data['ID'][pre:suf], y_pred[pre:suf], y_true[pre:suf]):
                        if tag == 'predict_clf':
                            res_scores[p[0]] = (y, t)
                        else:
                            if p[0] not in res_scores:
                                res_scores[p[0]] = {}
                            res_scores[p[0]][p[1]] = (y, t)

                num_valid += len(list_counts) - 1
            else:
                for k, eval_func in eval_metrics.items():
                    res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                for p, y, t in zip(input_data['ID'], y_pred, y_true):
                    if p[0] not in res_scores:
                        res_scores[p[0]] = {}
                    res_scores[p[0]][p[1]] = (y[1], t[1])
                num_valid += 1
        generator.reset()

        if tag in output_conf:
            save_path = output_conf[tag]['save_path'] + '-' + str(seed)

            if output_conf[tag]['save_format'] == 'TREC':
                with open(save_path, 'w') as f:
                    if tag == 'predict_clf':
                        for qid, entry in res_scores.items():
                            print >> f, '%s\t%d\t%d'%(qid, entry[0], entry[1])
                    else:
                        for qid, dinfo in res_scores.items():
                            dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                            for inum,(did, (score, gt)) in enumerate(dinfo):
                                print >> f, '%s\tQ0\t%s\t%d\t%f\t%s\t%s'%(qid, did, inum, score, config['net_name'], gt)
            elif output_conf[tag]['save_format'] == 'TEXTNET':
                with open(save_path, 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            print >> f, '%s %s %s %s'%(gt, qid, did, score)

        print '[Predict] results: ', '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()])
        sys.stdout.flush()


def main(argv):
    parser = argparse.ArgumentParser()
    # python main_conversation_qa.py --help to print the help messages
    # sys.argv includes a list of elements starting with the program
    # required parameters
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.', required=True)
    parser.add_argument('--model_file', default='./models/arci.config', help='Model_file: MatchZoo model file for the chosen model.', required=True)
    parser.add_argument('--or_cmd', default=False,
                        help='or_cmd: whether want to override config parameters by command line parameters', required=True)

    # optional parameters
    parser.add_argument('--gpu_id', help='gpu_id: Specifies which gpu to use by id')
    parser.add_argument('--embed_size', help='Embed_size: number of dimensions in word embeddings.')
    parser.add_argument('--embed_path', help='Embed_path: path of embedding file.')
    parser.add_argument('--test_relation_file', help='test_relation_file: path of test relation file.')
    parser.add_argument('--predict_relation_file', help='predict_relation_file: path of predict relation file.')
    parser.add_argument('--train_relation_file', help='train_relation_file: path of train relation file.')
    parser.add_argument('--valid_relation_file', help='valid_relation_file: path of valid relation file.')
    parser.add_argument('--vocab_size', help='vocab_size: vocab size')
    parser.add_argument('--text1_corpus', help='text1_corpus: path of text1 corpus')
    parser.add_argument('--text2_corpus', help='text2_corpus: path of text2 corpus')
    parser.add_argument('--weights_file', help='weights_file: path of weights file')
    parser.add_argument('--save_path', help='save_path: path of predicted score file')
    parser.add_argument('--valid_batch_list', help='valid_batch_list: batch size in valid data')
    parser.add_argument('--test_batch_list', help='test_batch_list: batch size in test data')
    parser.add_argument('--predict_batch_list', help='predict_batch_list: batch size in test data')
    parser.add_argument('--train_batch_size', help='train_batch_size: batch size in train data')
    parser.add_argument('--text1_max_utt_num', help='text1_max_utt_num: max number of utterances in dialog context')
    parser.add_argument('--cross_matrix', help='cross_matrix: parameters for model abalation')
    parser.add_argument('--inter_type', help='inter_type: parameters for model abalation')
    parser.add_argument('--test_weights_iters', help='test_weights_iters: the iteration of test weights file used')
    parser.add_argument('--num_iters')
    parser.add_argument('--keras_random_seed')
    parser.add_argument('--predict')
    parser.add_argument('--learning_rate', help='Set the learning rate')
    parser.add_argument('--use_existing_weights', help='Specify whether to use existing weights or overwrite')

    args = parser.parse_args()
    # parse the hyper-parameters from the command lines
    phase = args.phase
    model_file =  args.model_file
    or_cmd = bool(args.or_cmd)

    # load settings from the config file
    # then update the hyper-parameters in the config files with the settings passed from command lines
    with open(model_file, 'r') as f:
        config = json.load(f)
    if or_cmd:
        embed_size = args.embed_size
        embed_path = args.embed_path
        test_relation_file = args.test_relation_file
        predict_relation_file = args.predict_relation_file
        train_relation_file = args.train_relation_file
        valid_relation_file = args.valid_relation_file
        vocab_size = args.vocab_size
        text1_corpus = args.text1_corpus
        text2_corpus = args.text2_corpus
        weights_file = args.weights_file
        save_path = args.save_path
        text1_max_utt_num = args.text1_max_utt_num
        valid_batch_list = args.valid_batch_list
        predict_batch_list = args.predict_batch_list
        test_batch_list = args.test_batch_list
        train_batch_size = args.train_batch_size
        cross_matrix = args.cross_matrix
        inter_type = args.inter_type
        test_weights_iters = args.test_weights_iters
        gpu_id = args.gpu_id
        num_iters = args.num_iters
        keras_random_seed = args.keras_random_seed
        predict_eval = args.predict
        learning_rate = args.learning_rate
        use_existing_weights = args.use_existing_weights

        if embed_size != None:
            config['inputs']['share']['embed_size'] = int(embed_size)
        if embed_path != None:
            config['inputs']['share']['embed_path'] = embed_path
        if cross_matrix != None:
            config['inputs']['share']['cross_matrix'] = cross_matrix
        if inter_type != None:
            config['inputs']['share']['inter_type'] = inter_type
        if test_relation_file != None:
            config['inputs']['test']['relation_file'] = test_relation_file
        if predict_relation_file != None:
            config['inputs']['predict']['relation_file'] = predict_relation_file
        if train_relation_file != None:
            config['inputs']['train']['relation_file'] = train_relation_file
        if valid_relation_file != None:
            config['inputs']['valid']['relation_file'] = valid_relation_file
        if vocab_size != None:
            config['inputs']['share']['vocab_size'] = int(vocab_size)
        if text1_corpus != None:
            config['inputs']['share']['text1_corpus'] = text1_corpus
        if text2_corpus != None:
            config['inputs']['share']['text2_corpus'] = text2_corpus
        if weights_file != None:
            config['global']['weights_file'] = weights_file
        if save_path != None:
            config['outputs']['predict']['save_path'] = save_path
        if text1_max_utt_num != None:
            config['inputs']['share']['text1_max_utt_num'] = int(text1_max_utt_num)
        if valid_batch_list != None:
            config['inputs']['valid']['batch_list'] = int(valid_batch_list)
        if test_batch_list != None:
            config['inputs']['test']['batch_list'] = int(test_batch_list)
        if predict_batch_list != None:
            config['inputs']['predict']['batch_list'] = int(predict_batch_list)
        if train_batch_size != None:
            config['inputs']['train']['batch_size'] = int(train_batch_size)
        if test_weights_iters != None:
            config['global']['test_weights_iters'] = int(test_weights_iters)
        if num_iters != None:
            config['global']['num_iters'] = int(num_iters)
        if learning_rate != None:
            config['global']['learning_rate'] = float(learning_rate)
        if use_existing_weights != None:
            config['global']['use_existing_weights'] = use_existing_weights
        if keras_random_seed != None:
            global seed
            seed = int(keras_random_seed)
            random.seed(seed)
            numpy.random.seed(seed)
            tensorflow.set_random_seed(seed)
        if predict_eval != None:
            config['inputs']['share']['predict'] = predict_eval

        if gpu_id is not None and gpu_id.isdigit():
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    if phase == 'train':
        train(config)
    elif phase == 'predict':
        predict(config)
    else:
        print 'Phase Error.'
    return


if __name__=='__main__':
    main(sys.argv)
