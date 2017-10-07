import json
import sys
import argparse
import os
import numpy as np
from utils import set_tag, tag2answer, tag_correct, y2answer
import rnn_model
import match_lstm_model
from evaluate import evaluate
import tensorflow as tf
import time

epoch = 0
max_em = -1
saving_path = None

def train(args, sess, model, train_data, eval_test=None, saver=None):
    global epoch, max_em, saving_path
    total_nll = 0
    tic = time.time()
    for batch in range(0,len(train_data)):
        x_data,cx_data,q_data,cq_data,tag_data,id_data,y_data,ori_x_data,answer_text_data = train_data[batch]
        acc = 0.
        if args.model == "seq_tag" or args.model == "match_lstm":
            feed_dict = {model.context:x_data, model.query:q_data, model.word_tags:tag_data}
            _,nll,acc = sess.run([model.train, model.nll,model.acc], feed_dict=feed_dict)
        elif args.model == "crf" or args.model == "match_lstm_crf":
            nll,acc = model.train(sess, x_data, q_data, tag_data)
        elif args.model == "pointer" or args.model == "match_lstm_pointer":
            y1 = np.array([y[0][1] for y in y_data])
            y2 = np.array([y[1][1]-1 for y in y_data])      #y[1][1] states the next of end word
            feed_dict = {model.context:x_data, model.query:q_data, model.y1:y1,model.y2:y2}
            _,nll = sess.run([model.train, model.nll], feed_dict=feed_dict)
        total_nll += nll
        if args.verbose:
            print('[learning] >> %2.2f%% nll %f accuracy %f'%(batch*100./len(train_data),nll,acc),'completed in %.2f (sec) <<\r'%(time.time()-tic))
            sys.stdout.flush()
        if eval_test != None and batch % args.evaluate == 0 and batch > 0:
            e = eval_test()
            print('training batch %d, f1 %f, exact match %f'%(batch, e['f1'], e['exact_match']))
            sys.stdout.flush()
            if saver != None and max_em != None:
                if max_em < e['exact_match']:
                    max_em = e['exact_match']
                    saving_path = saver.save(sess, './models/'+args.saving_path)
            sys.stdout.flush()
    return total_nll
        
def test(args, sess, model, data):
    total_tags = {}
    for batch in range(0,len(data)):
        x_data,cx_data,q_data,cq_data,tag_data,id_data,y_data,ori_x,answer_text_data = data[batch]
        if args.model == "seq_tag" or args.model == "match_lstm":
            feed_dict = {model.context:x_data, model.query:q_data, model.word_tags:tag_data}
            pred_tags, probs = sess.run([model.pred_tags,model.probs], feed_dict=feed_dict)
            for xi, ori_xi in zip(x_data,ori_x):
                assert np.sum([word>0 for word in xi]) == len(ori_xi), ('mapped x and original x not compatable')
            answers = [tag2answer(ori_xi,xi,tag,prob,tag_num=args.tag_num) for ori_xi,xi,tag,prob in zip(ori_x,x_data,pred_tags,probs)]
            #answers = [tag2answer(ori_xi,xi,tag,prob,tag_num=args.tag_num) for ori_xi,xi,tag,prob in zip(ori_x,x_data,tag_data,np.ones(x_data.shape))]
            if args.mode == 'inference':
                for answer, real_answer, xi, real_tag, tag in zip(answers, answer_text_data, ori_x, tag_data, pred_tags):
                    print(' '.join(xi))
                    print('real answer',real_answer)
                    print('predicted answer',answer)
                    #print(' '.join([str(num) for num in real_tag]))
                    print(' '.join([str(num) for num in tag]))
                    sys.stdout.flush()
        elif args.model == "crf" or args.model == "match_lstm_crf":
            pred_tags, probs = model.inference(sess, x_data, q_data)
            answers = [tag2answer(ori_xi,xi,tag,prob,tag_num=args.tag_num) for ori_xi,xi,tag,prob in zip(ori_x,x_data,pred_tags,probs)]
            #answers = [tag2answer(ori_xi,xi,tag,prob,tag_num=args.tag_num) for ori_xi,xi,tag,prob in zip(ori_x,x_data,tag_data,np.ones(x_data.shape))]
            if args.mode == 'inference':
                for answer, real_answer, xi, tag in zip(answers, answer_text_data, ori_x, pred_tags):
                    print(' '.join(xi))
                    print('real answer',real_answer)
                    print('predicted answer',answer)
                    print(' '.join([str(num) for num in tag]))
                    sys.stdout.flush()
        elif args.model == "pointer" or args.model == "match_lstm_pointer":
            y1 = np.array([y[0][1] for y in y_data])
            y2 = np.array([y[1][1]-1 for y in y_data])
            '''
            y1_data = np.zeros((len(y1),len(x_data[0,:])))
            y2_data = np.zeros((len(y2),len(x_data[0,:])))
            y1_data[np.arange(len(y1)),y1] = 1.
            y2_data[np.arange(len(y2)),y2] = 1.
            '''
            feed_dict = {model.context:x_data, model.query:q_data, model.y1:y1,model.y2:y2}
            y1,y2 = sess.run([model.pred_y1, model.pred_y2], feed_dict = feed_dict)
            answers = [y2answer(yy1,yy2,xi) for yy1,yy2,xi in zip(y1,y2,ori_x)]
            if args.mode == 'inference':
                for answer, real_answer, xi in zip(answers, answer_text_data, ori_x):
                    print(' '.join(xi))
                    print('real answer',real_answer)
                    print('predicted answer',answer)
                    sys.stdout.flush()
        for id, answer in zip(id_data, answers):
            total_tags[id] = answer
    return total_tags

    
def load(args, data_type):
    # data_type is either train or dev or test
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    data = json.load(open(data_path))
    shared = json.load(open(shared_path))
    return data, shared

def word_mapping_and_padding(words, word2idx, maxlen):
    result = word_mapping(words, word2idx)
    while len(result) < maxlen:
        result.append(word2idx['PADDING'])
    return np.array(result,dtype=np.int32)

def word_mapping(words, word2idx):
    result = []
    for word in words:
        if word in word2idx:
            result.append(word2idx[word])
        elif word.lower() in word2idx:
            result.append(word2idx[word.lower()])
        else:
            result.append(word2idx['OOV'])
    return result

def char_mapping_and_padding(chars, char2idx, maxlen):
    result = char_mapping(chars, char2idx)
    while len(result) < maxlen:
        result.append(char2idx['PADDING'])
    return np.array(result,dtype=np.int32)

def char_mapping(chars, char2idx):
    result = []
    for char in chars:
        if char in char2idx:
            result.append(char2idx[char])
        elif char.lower() in char2idx:
            result.append(char2idx[char.lower()])
        else:
            result.append(char2idx['OOV'])
    return result

def tag_mapping_and_padding(tags, maxlen, tag_num):
    result = tag_mapping(tags, tag_num)
    while len(result) < maxlen:
        result.append(0)
    return np.array(result)

def tag_mapping(tags, tag_num):
    assert tag_num > 2 and tag_num <= 6
    if tag_num == 6:
        tag_dict = {'PADDING':0,'O':1,'B':2,'I':3,'E':4,'S':5}
    elif tag_num == 5:
        tag_dict = {'PADDING':0,'O':1,'B4':2,'I4':3,'E4':4}
    elif tag_num == 4:
        tag_dict = {'PADDING':0,'O':1,'B3':2,'I3':3}
    elif tag_num == 3:
        tag_dict = {'PADDING':0,'O':1,'Y':2}
    assert len(tag_dict) == tag_num
    return [tag_dict[t] for t in tags]

def build_batches(args, data, shared_data, data_type):
    q = data['q']
    cq = data['cq']
    rx = data['*x']
    x = shared_data['x']
    cx = shared_data['cx']
    y = data['y']
    ids = data['ids']
    answerss = data['answerss']
    data_batches = {}
    for rxi,qi,cqi,yi,answers,id in zip(rx,q,cq,y,answerss,ids):
        # get context
        xi = x[rxi[0]][rxi[1]][0]
        cxi = cx[rxi[0]][rxi[1]][0]
        for j,yij,answer in zip(range(len(yi)),yi,answers):
            assert len(yij) == 2, (yij,yi)
            assert len(yij[0]) == 2, yij
            tag = set_tag(xi, yij[0],yij[1])
            data_batches[id+'_'+str(j)] = {'context':xi,'question':qi,'tags':tag,'answer_text':answer}
    json.dump(data_batches, open('{}_tagged_data.json'.format(data_type),'w'))

def make_batches(args, data, shared, word2idx, char2idx, tag_num):
    x = shared['x']
    cx = shared['cx']
    bs = args.batch_size
    batches = []
    down_sample = 1
    if args.debug:
        down_sample = 1000
    print('making batches')
    sys.stdout.flush()
    for batch in range(0,len(data)//down_sample,bs):
        rx_batch = [d[3] for d in data[batch:batch+bs]]
        q_batch = [d[1] for d in data[batch:batch+bs]]
        cq_batch = [d[2] for d in data[batch:batch+bs]]
        y_batch = [d[4] for d in data[batch:batch+bs]]
        id_batch = [d[5] for d in data[batch:batch+bs]]
        answer_text_batch = [d[6] for d in data[batch:batch+bs]]
        batches.append(make_batch(x, cx, rx_batch, q_batch, cq_batch, y_batch, id_batch, word2idx, char2idx, tag_num, answer_text_batch))
    return batches
        

def make_batch(x, cx, rxis, qis, cqis, yis, ids, word2idx, char2idx, tag_num, answer_text_batch):
    assert len(rxis) == len(qis) and len(qis) == len(yis), (len(rxis),len(qis),len(yis))
    ori_x_data = []
    x_data = []
    cx_data = []
    q_data = []
    cq_data = []
    tag_data = []
    y_data = []
    context_max_len = 0
    query_max_len = 0
    c_max_len = 0
    for rxi,qi,cqi in zip(rxis,qis,cqis):
        xi = x[rxi[0]][rxi[1]][0]
        cxi = cx[rxi[0]][rxi[1]][0]
        context_max_len = max(context_max_len, len(xi))
        query_max_len = max(query_max_len, len(qi))
        c_max_len = max([c_max_len, max([len(word) for word in cxi]), max([len(word) for word in cqi])])

    for rxi,qi,cqi,yi in zip(rxis,qis,cqis,yis):
        # get context
        xi = x[rxi[0]][rxi[1]]
        cxi = cx[rxi[0]][rxi[1]]
        yij = yi[0]
        #for yij in yi:
        tag = set_tag(xi, yij[0],yij[1],tag_num)
        #print('tags',tag[0])
        # here needs to be noticed, sentence splitting is not applied
        #x_data.append(np.array([word_mapping_and_padding(words,word2idx,context_max_len) for words in xi][0],dtype=np.int32))
        ori_x_data.append(xi[0])
        x_data.append(word_mapping_and_padding(xi[0],word2idx,context_max_len))
        cx_data.append(np.array([char_mapping_and_padding(word,char2idx,c_max_len) for word in cxi[0]],dtype=np.int32))
        q_data.append(np.array(word_mapping_and_padding(qi,word2idx,query_max_len),dtype=np.int32))
        cq_data.append(np.array([char_mapping_and_padding(word,char2idx,c_max_len) for word in cqi],dtype=np.int32))
        #tag_data.append([np.array(tag_mapping_and_padding(t,context_max_len)) for t in tag][0])
        tag_data.append(np.array(tag_mapping_and_padding(tag[0],context_max_len, tag_num)))      # because there is only one sentence in the context
        #print('mapped tag',' '.join([str(num) for num in tag_data[-1]]))
        y_data.append(yij)
    return (np.array(x_data),np.array(cx_data),np.array(q_data),np.array(cq_data),np.array(tag_data),ids,y_data,ori_x_data, answer_text_batch)

        
def build_voc(args, word2vec_dict, char_counter):
    word2idx = {'PADDING':0,'OOV':1}
    idx2word = ['PADDING','OOV']
    char2idx = {'PADDING':0,'OOV':1}
    idx2char = ['PADDING','OOV']
    vec = [np.zeros(args.glove_vec_size),np.random.normal(size=(args.glove_vec_size,),scale=0.1)]
    for word in word2vec_dict:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
            idx2word.append(word)
            vec.append(np.array(word2vec_dict[word],dtype=np.float32))
    for char in char_counter:
        if char not in char2idx:
            char2idx[char] = len(vec)
            idx2char.append(char)
            vec.append(np.random.normal(size=(args.glove_vec_size,),scale=0.1))
    return word2idx, idx2word, char2idx, idx2char, np.array(vec,dtype=np.float32)
    
def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/squad"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-s','--saving_path')
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument('-v',"--verbose",action='store_true')
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=300, type=int)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument('-m',"--model",default="seq_tag",type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--split", action='store_true')
    parser.add_argument("--batch_size",default=20,type=int)
    parser.add_argument("--hidden_size",default=200,type=int)
    parser.add_argument("--tag_num",default=6,type=int)
    parser.add_argument("--epoch",default=20,type=int)
    parser.add_argument("--evaluate",default=500,type=int)
    # TODO : put more args here
    return parser.parse_args()

def data_sort(data, shared):
    print('sorting data')
    sys.stdout.flush()
    q = data['q']
    cq = data['cq']
    rx = data['*x']
    y = data['y']
    ids = data['ids']
    answer_text = data['answerss']
    x = shared['x']
    data = [(len(x[rxi[0]][rxi[1]][0]),qi,cqi,rxi,yi,id,at) for qi,cqi,rxi,yi,id,at in zip(q,cq,rx,y,ids,answer_text)]
    data.sort(key=lambda i:i[0])
    return data


def process(args):
    if args.mode == 'train' or args.mode == 'inference':
        train_data, train_shared_data = load(args, 'train')
        train_data = data_sort(train_data, train_shared_data)
        dev_data, dev_shared_data = load(args, 'dev')
        dev_data = data_sort(dev_data, dev_shared_data)
        word2vec = train_shared_data['word2vec']
        word2vec.update(dev_shared_data['word2vec'])
        word2idx, idx2word, char2idx, idx2char, vec = build_voc(args, word2vec,train_shared_data['char_counter'])
        train_data = make_batches(args, train_data, train_shared_data, word2idx, char2idx, args.tag_num)
        dev_data = make_batches(args, dev_data, dev_shared_data, word2idx, char2idx, args.tag_num)
        dev_data_path = os.path.join(args.source_dir, "dev-v1.1.json")
        dev_data_set = json.load(open(dev_data_path))
        gold_dataset = dev_data_set['data']
        model = None
        if args.model == 'seq_tag':
            model = rnn_model.Basic_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        elif args.model == 'crf':
            model = rnn_model.CRF_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        elif args.model == 'match_lstm':
            model = match_lstm_model.Match_LSTM_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        elif args.model == 'pointer':
            model = rnn_model.Pointer_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        elif args.model == 'match_lstm_pointer':
            model = match_lstm_model.Pointer_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        elif args.model == 'match_lstm_crf':
            model = match_lstm_model.CRF_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            eval_test = lambda: evaluate(gold_dataset, test(args, sess, model, dev_data))
            if args.mode == 'train':
                max_f1 = 0
                global max_em, epoch, saving_path
                while epoch < args.epoch:
                    total_nll = train(args, sess, model, train_data, eval_test, saver)
                    e = eval_test()
                    f1 = e['f1']
                    em = e['exact_match']
                    print('epoch %d, f1 score %f, exact match %f'%(epoch, f1, em))
                    sys.stdout.flush()
                    if em > max_em:
                        saving_path = saver.save(sess, './models/'+args.saving_path)
                        print('saving_path',saving_path)
                        sys.stdout.flush()
                        max_em = em 
                        max_f1 = f1
                    epoch += 1
                print('max f1',max_f1,'max exact match',max_em)
                sys.stdout.flush()
            if args.mode == 'inference':
                saving_path = './models/'+args.saving_path
            saver.restore(sess, saving_path)
            #print(tf.get_default_graph())
            args.mode = 'inference'
            e = eval_test()
            f1 = e['f1']
            em = e['exact_match']
            print('saving path', saving_path)
            print('restore result, f1 score %f, exact match %f'%(f1, em))
            sys.stdout.flush()

    elif args.mode == 'output_tagged':
        train_data, train_shared_data = load(args, 'train')
        build_batches(args, train_data, train_shared_data, 'train')
        dev_data, dev_shared_data = load(args, 'dev')
        build_batches(args, dev_data, dev_shared_data, 'dev')

    '''
    elif args.mode == 'inference':
        print('inference mode')
        train_data, train_shared_data = load(args, 'train')
        train_data = data_sort(train_data, train_shared_data)
        dev_data, dev_shared_data = load(args, 'dev')
        dev_data = data_sort(dev_data, dev_shared_data)
        word2vec = train_shared_data['word2vec']
        word2vec.update(dev_shared_data['word2vec'])
        word2idx, idx2word, char2idx, idx2char, vec = build_voc(args, word2vec,train_shared_data['char_counter'])
        train_data = make_batches(args, train_data, train_shared_data, word2idx, char2idx, args.tag_num)
        dev_data = make_batches(args, dev_data, dev_shared_data, word2idx, char2idx, args.tag_num)
        dev_data_path = os.path.join(args.source_dir, "dev-v1.1.json")
        dev_data_set = json.load(open(dev_data_path))
        gold_dataset = dev_data_set['data']
        if args.model == 'seq_tag':
            model = rnn_model.Basic_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        elif args.model == 'crf':
            model = rnn_model.CRF_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        elif args.model == 'match_lstm':
            model = match_lstm_model.Match_LSTM_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        elif args.model == 'pointer':
            model = rnn_model.Pointer_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        elif args.model == 'match_lstm_pointer':
            model = match_lstm_model.Pointer_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        elif args.model == 'match_lstm_crf':
            model = match_lstm_model.CRF_Model(vec, args.glove_vec_size, args.hidden_size, args.tag_num)
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            eval_test = lambda: evaluate(gold_dataset, test(args, sess, model, dev_data))
            saving_path = args.saving_path
            saver.restore(sess, saving_path)
            e = eval_test()
            f1 = e['f1']
            em = e['exact_match']
            print('inference f1 %f exact match %f'%(f1,em))
            sys.stdout.flush()
    '''
        
        

def main():
    args = get_args()
    process(args)
    

if __name__ == '__main__':
    main()
