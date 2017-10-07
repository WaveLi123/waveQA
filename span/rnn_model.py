import tensorflow as tf
import numpy as np

class Basic_Model:
    def __init__(self, vec, de, nh, nt):
        self._build_parameter(vec, de, nh, nt)
        self._build_inputs()
        query_vec, query_hidden = self._build_query()
        context_vec, context_hidden = self._build_context()
        output_hidden = self._inference(context_hidden=context_hidden, context_vec=context_vec, query_hidden = query_hidden,query_vec=query_vec)
        self.logits = self._output(output_hidden)
        self.nll = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.boolean_mask(self.word_tags,self.tag_mask),logits=tf.boolean_mask(self.logits,self.context_mask)))
        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.nll)
        self.pred_tags = tf.argmax(self.logits,2)*tf.to_int64(self.tag_mask)
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
        self.tag_mask = tf.greater(self.word_tags,0)

    def _build_context(self):
        with tf.variable_scope('context') as scope:
            cell_f = tf.contrib.rnn.GRUCell(num_units=self.nh)
            cell_b = tf.contrib.rnn.GRUCell(num_units=self.nh)
            with tf.device('/cpu:0'): x = tf.nn.embedding_lookup(self.emb, self.context)
            outputs,state = tf.nn.bidirectional_dynamic_rnn(cell_f,cell_b,inputs=x,sequence_length=self.context_length,dtype=tf.float32,time_major=False)
            return tf.concat(state,-1), tf.concat(outputs,2)

    def _build_query(self):
        with tf.variable_scope('query') as scope:
            cell_f = tf.contrib.rnn.GRUCell(num_units=self.nh)
            cell_b = tf.contrib.rnn.GRUCell(num_units=self.nh)
            with tf.device('/cpu:0'): x = tf.nn.embedding_lookup(self.emb, self.query)
            outputs,state = tf.nn.bidirectional_dynamic_rnn(cell_f,cell_b,inputs=x,sequence_length=self.query_length,dtype=tf.float32,time_major=False)
            return tf.concat(state,-1), tf.concat(outputs, 2)

    def _attention(self, query_hidden, context_hidden, scope_name='attention'):
        with tf.variable_scope(scope_name) as scope:
            '''
            # traditional attention
            att_w1 = tf.get_variable('att_w1',(self.nh*2,self.nh*2),initializer=tf.random_normal_initializer(stddev=0.1))
            att_w2 = tf.get_variable('att_w2',(self.nh*2,self.nh*2),initializer=tf.random_normal_initializer(stddev=0.1))
            att_u = tf.get_variable('att_u',(self.nh*2,1),initializer=tf.random_normal_initializer(stddev=0.1))
            att_b = tf.get_variable('att_b',(self.nh*2,),initializer=tf.random_normal_initializer(stddev=0.1))
            t_query_hidden = tf.transpose(query_hidden,[1,0,2])
            def _att(c_vec):
                #h = tf.tanh(tf.expand_dims(tf.matmul(c_vec, att_w1),1) + tf.tensordot(query_hidden, att_w2,[[-1],[0]]) + att_b)
                h = tf.transpose(tf.map_fn(fn=lambda x: tf.tanh(tf.matmul(c_vec, att_w1)+tf.matmul(x, att_w2)+att_b), elems=t_query_hidden),[1,0,2])
                p = tf.nn.softmax(tf.tensordot(h, att_u, [[-1],[0]]), 1)
                return tf.reduce_sum(p*query_hidden, 1)
            '''
            memory_context = tf.transpose(tf.map_fn(lambda x: self._luong_attention(x, query_hidden, scope_name), tf.transpose(context_hidden, [1,0,2])),[1,0,2])
            return memory_context

    def _luong_attention(self, x, memory, scope_name='luong'):
        '''
            x (bs, nh); memory (bs, sl, nh)
        '''
        with tf.variable_scope(scope_name) as scope:
            score_w  = tf.get_variable('score_w', (self.nh*2, self.nh*2) , initializer=tf.random_normal_initializer(stddev=0.1))
            h = tf.expand_dims(tf.matmul(x, score_w),1)
            score = tf.nn.softmax(tf.reduce_sum(memory*h, 2),1)
            c = tf.reduce_sum(tf.expand_dims(score,2)*memory, 1)
            output_w = tf.get_variable('output_w',(self.nh*4,self.nh*2),initializer=tf.random_normal_initializer(stddev=0.1))
            return tf.tanh(tf.matmul(tf.concat([c, x], -1),output_w))
            


    def _inference(self, context_hidden, context_vec, query_hidden, query_vec):
        with tf.variable_scope('inference') as scope:
            cell_f = tf.contrib.rnn.GRUCell(num_units=self.nh*2)
            cell_b = tf.contrib.rnn.GRUCell(num_units=self.nh*2)
            '''
            query_att = self._attention(query_hidden, context_hidden, 'question_att')
            context_att = self._attention(context_hidden, query_att, 'context_self_att')
            '''
            w_ff_1 = tf.get_variable('w_ff_1',(self.nh*2,self.nh*2),initializer=tf.random_normal_initializer(stddev=0.1))
            w_ff_2 = tf.get_variable('w_ff_2',(self.nh*2,self.nh*2),initializer=tf.random_normal_initializer(stddev=0.1))
            x = tf.nn.relu(tf.tensordot(context_hidden, w_ff_1,[[-1],[0]]) + tf.expand_dims(tf.matmul(query_vec, w_ff_2),1))
            hidden,state = tf.nn.bidirectional_dynamic_rnn(cell_f,cell_b,inputs=x,sequence_length=self.context_length,dtype=tf.float32,time_major=False)
            return tf.concat(hidden, 2)

    def _output(self, hidden):
        with tf.variable_scope('output') as scope:
            w_s = tf.get_variable('w_s',(self.nh*4,self.nt),initializer=tf.random_normal_initializer(stddev=0.1))
            return tf.tensordot(hidden, w_s, [[-1],[0]])
        

class CRF_Model(Basic_Model):
    def __init__(self, vec, de, nh, nt):
        super(CRF_Model, self)._build_parameter(vec, de, nh, nt)
        super(CRF_Model, self)._build_inputs()
        context_vec,context_hidden = super(CRF_Model,self)._build_context()
        query_vec, query_hidden = super(CRF_Model,self)._build_query()
        output_hidden = super(CRF_Model,self)._inference(context_hidden,context_vec,query_hidden, query_vec)
        self.unary_scores = super(CRF_Model, self)._output(output_hidden)
        self.transition_params = tf.get_variable('transitions', [nt,nt], initializer=tf.random_normal_initializer(stddev=0.1))
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.unary_scores, self.word_tags, self.context_length, transition_params=self.transition_params)
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
            #tag_seq.append(np.pad(viterbi_sequence,(0,pad_length-len(viterbi_sequence)),mode='constant',constant_values=0))
            tag_seq.append(np.concatenate([viterbi_sequence,np.zeros((pad_length-len(viterbi_sequence)))],0))
            assert len(tag_seq[-1]) == pad_length, ('padding error',len(tag_seq[-1]),pad_length)
        return tag_seq, np.amax(tf_unary_scores, 2)

class Pointer_Model(Basic_Model):
    def __init__(self, vec, de, nh, nt):
        super(Pointer_Model, self)._build_parameter(vec, de, nh, nt)
        self._build_inputs()
        context_vec, context_hidden = super(Pointer_Model,self)._build_context()
        query_vec, query_hidden = super(Pointer_Model,self)._build_query()
        output_hidden = super(Pointer_Model,self)._inference(context_hidden,context_vec,query_hidden, query_vec)
        self.d1, self.d2 = self._output(output_hidden, query_vec)
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

    def _output(self, context_hidden, query_vec):
        with tf.variable_scope('inference') as scope:
            w1 = tf.get_variable('w1',(self.nh*6,self.nh*2),initializer=tf.random_normal_initializer(stddev=0.1))
            w2 = tf.get_variable('w2',(self.nh*8,self.nh*2),initializer=tf.random_normal_initializer(stddev=0.1))
            ws = tf.get_variable('ws',(self.nh*2,1),initializer=tf.random_normal_initializer(stddev=0.1))
            h1 = tf.nn.relu(tf.transpose(tf.map_fn(fn=lambda x:tf.matmul(tf.concat([x,query_vec],-1),w1),elems=tf.transpose(context_hidden,[1,0,2])),[1,0,2]))    # (bs, sl, 2*nh)
            d1 = tf.nn.softmax(tf.squeeze(tf.tensordot(h1,ws,[[-1],[0]])))     # bs,sl,1
            h2 = tf.nn.relu(tf.transpose(tf.map_fn(fn=lambda x:tf.matmul(tf.concat([x,query_vec],-1),w2),elems=tf.transpose(tf.concat([context_hidden,h1],2),[1,0,2])),[1,0,2]))
            d2 = tf.nn.softmax(tf.squeeze(tf.tensordot(h2,ws,[[-1],[0]])))     # bs,sl,1
        return d1, d2
