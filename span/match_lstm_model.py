from rnn_model import Basic_Model
from my.rnn import match_bidirectional_dynamic_rnn, match_dynamic_rnn
from my.my_cell import MatchCell
import tensorflow as tf
import numpy as np

class Match_LSTM_Model:
    def __init__(self, vec, de, nh, nt):
        self._build_parameter(vec, de, nh, nt)
        self._build_inputs()
        p_context_vec = self._build_context(self.context)
        q_context_vec = self._build_context(self.query, reuse=True)
        H = self._match(Vq=q_context_vec, Vp=p_context_vec)
        self.logits = self._inference(H)

        self.nll = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.boolean_mask(self.word_tags,self.context_mask),logits=tf.boolean_mask(self.logits,self.context_mask)))
        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.nll)
        self.pred_tags = tf.argmax(self.logits,2)*tf.to_int64(self.context_mask)
        self.probs = tf.reduce_max(tf.nn.softmax(self.logits),2)
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(self.pred_tags), self.word_tags)))


    def _build_parameter(self, vec, de, nh, nt):
        with tf.device('/cpu:0'): self.emb = tf.Variable(vec,dtype=tf.float32,name='emb')
        self.de = de
        self.nh = nh
        self.nt = nt

    def _build_inputs(self):
        self.context = tf.placeholder(tf.int32,shape=(None,None),name='context')
        self.context_mask = tf.greater(self.context,0)
        self.context_length = tf.reduce_sum(tf.to_int32(self.context_mask),-1)
        self.query = tf.placeholder(tf.int32, shape=(None,None),name='query')
        self.query_mask = tf.greater(self.query,0)
        self.query_length = tf.reduce_sum(tf.to_int32(self.query_mask),-1)
        self.word_tags = tf.placeholder(tf.int32, shape=(None,None),name='tags')

    def _build_context(self, idxs, scope_name='context',reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse: scope.reuse_variables()
            cell_f = tf.contrib.rnn.GRUCell(num_units=self.nh)
            cell_b = tf.contrib.rnn.GRUCell(num_units=self.nh)
            sent_length = tf.reduce_sum(tf.to_int32(tf.greater(idxs,0)),-1)
            with tf.device('/cpu:0'): x = tf.nn.embedding_lookup(self.emb, idxs)
            outputs,state = tf.nn.bidirectional_dynamic_rnn(cell_f,cell_b,inputs=x,sequence_length=sent_length,dtype=tf.float32,time_major=False)
            return tf.concat(outputs,2)

    def _match(self, Vq, Vp):
        with tf.variable_scope('match') as scope:
            #cell_f = tf.contrib.rnn.GRUCell(num_units=self.nh)
            #cell_b = tf.contrib.rnn.GRUCell(num_units=self.nh)
            cell_f = MatchCell(num_units=self.nh)
            cell_b = MatchCell(num_units=self.nh)
            (output_fw, output_bw), (state_fw,state_bw) = match_bidirectional_dynamic_rnn(cell_f, cell_b, inputs=Vp, Hq=Vq, sequence_length=self.context_length, dtype=tf.float32,time_major=False)
            output = tf.concat([output_fw, output_bw], -1)
            return output

    def _inference(self, H):
        with tf.variable_scope('inference') as scope:
            w_s = tf.get_variable('w_s',(self.nh*2,self.nt),initializer=tf.random_normal_initializer(stddev=0.1))
            cell = tf.contrib.rnn.GRUCell(num_units=self.nh*2)
            output, state = tf.nn.dynamic_rnn(cell, inputs=H, sequence_length=self.context_length, dtype=tf.float32,time_major=False)
            h = tf.tensordot(output, w_s, [[-1],[0]])
            return h


class Pointer_Model(Match_LSTM_Model):
    def __init__(self, vec, de, nh, nt):
        super(Pointer_Model, self)._build_parameter(vec, de, nh, nt)
        self._build_inputs()
        p_context_vec = super(Pointer_Model,self)._build_context(self.context)
        q_context_vec = super(Pointer_Model,self)._build_context(self.query, reuse=True)
        H = super(Pointer_Model,self)._match(Vq=q_context_vec, Vp=p_context_vec)
        self.d1, self.d2 = self._inference(H)
        self.d1 = self.d1*tf.to_float(self.context_mask)
        self.d2 = self.d2*tf.to_float(self.context_mask)

        self.nll = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y1,logits=self.d1)+tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y2,logits=self.d2))
        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.nll)
        self.pred_y1 = tf.argmax(self.d1,1)
        self.pred_y2 = tf.argmax(self.d2,1)

    def _build_inputs(self):
        self.context = tf.placeholder(tf.int32,shape=(None,None),name='context')
        self.context_mask = tf.greater(self.context,0)
        self.context_length = tf.reduce_sum(tf.to_int32(self.context_mask),-1)
        self.query = tf.placeholder(tf.int32, shape=(None,None),name='query')
        self.query_mask = tf.greater(self.query,0)
        self.query_length = tf.reduce_sum(tf.to_int32(self.query_mask),-1)
        self.y1 = tf.placeholder(tf.int32,shape=(None,), name='y1')
        self.y2 = tf.placeholder(tf.int32,shape=(None,), name='y2')

    def _inference(self, H):
        with tf.variable_scope('inference') as scope:
            w1 = tf.get_variable('w1',(self.nh*2,self.nh*2),initializer=tf.random_normal_initializer(stddev=0.1))
            w2 = tf.get_variable('w2',(self.nh*4,self.nh*2),initializer=tf.random_normal_initializer(stddev=0.1))
            ws = tf.get_variable('ws',(self.nh*2,1),initializer=tf.random_normal_initializer(stddev=0.1))
            h1 = tf.tanh(tf.transpose(tf.map_fn(fn=lambda x:tf.matmul(x,w1),elems=tf.transpose(H,[1,0,2])),[1,0,2]))    # (bs, sl, 2*nh)
            d1 = tf.squeeze(tf.tensordot(h1,ws,[[-1],[0]]))     # bs,sl,1
            h2 = tf.tanh(tf.transpose(tf.map_fn(fn=lambda x:tf.matmul(x,w2),elems=tf.transpose(tf.concat([H,h1],2),[1,0,2])),[1,0,2]))
            d2 = tf.squeeze(tf.tensordot(h2,ws,[[-1],[0]]))     # bs,sl,1
        return d1, d2

class CRF_Model(Match_LSTM_Model):
    def __init__(self, vec, de, nh, nt):
        super(CRF_Model, self)._build_parameter(vec, de, nh, nt)
        super(CRF_Model, self)._build_inputs()
        p_context_vec = super(CRF_Model,self)._build_context(self.context)
        q_context_vec = super(CRF_Model,self)._build_context(self.query, reuse=True)
        H = super(CRF_Model,self)._match(Vq=q_context_vec, Vp=p_context_vec)
        self.unary_scores = super(CRF_Model,self)._inference(H)
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.unary_scores, self.word_tags, self.context_length)
        self.crf_loss = tf.reduce_mean(-self.log_likelihood)
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.crf_loss)

    def train(self, sess, x, q, tags):
        feed_dict = {self.context:x, self.query:q, self.word_tags:tags}
        loss, _, tf_unary_scores, tf_sequence_lengths, tf_transition_params = sess.run([self.crf_loss, self.train_op, self.unary_scores, self.context_length, self.transition_params],feed_dict=feed_dict)
        acc_seq = []
        #pad_length = tf_unary_scores.shape[1]
        for tf_unary_scores_, tf_sequence_lengths_, tag in zip(tf_unary_scores, tf_sequence_lengths,tags):
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(tf_unary_scores_[:tf_sequence_lengths_], tf_transition_params)
            #viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(tf_unary_scores_, tf_transition_params)
            acc_seq.append(np.mean(viterbi_sequence==tag[:tf_sequence_lengths_]))
        return loss,np.mean(acc_seq)

    def inference(self, sess, x, q):
        tag_seq = []
        feed_dict = {self.context:x, self.query:q}
        tf_unary_scores, tf_sequence_lengths, tf_transition_params = sess.run([self.unary_scores, self.context_length, self.transition_params],feed_dict=feed_dict)
        pad_length = tf_unary_scores.shape[1]
        for tf_unary_scores_, tf_sequence_lengths_ in zip(tf_unary_scores, tf_sequence_lengths):
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(tf_unary_scores_[:tf_sequence_lengths_], tf_transition_params)
            #viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(tf_unary_scores_, tf_transition_params)
            #tag_seq.append(viterbi_sequence)
            tag_seq.append(np.pad(viterbi_sequence,(0,pad_length-len(viterbi_sequence)),mode='constant',constant_values=0))
            assert len(tag_seq[-1]) == pad_length, ('padding error',len(tag_seq[-1]),pad_length)
        return tag_seq, np.amax(tf_unary_scores, 2)

