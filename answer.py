import io

import sys

#改变标准输出的默认编码

sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.stdin=io.TextIOWrapper(sys.stdin.buffer,encoding='utf-8')

import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import word_token
import jieba
import difflib
from bottle import route,run,static_file
from whoosh.index import create_in,open_dir
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
class Answer:
    '''
        问答系统
    '''

    def __init__(self):
        self.size = 8               # LSTM神经元size
        self.GO_ID = 1              # 输出序列起始标记
        self.EOS_ID = 2             # 结尾标记
        self.PAD_ID = 0             # 空值填充0
        self.min_freq = 1           # 样本词频超过这个值才会存入词表
        self.epochs = 20000         # 训练次数
        self.batch_num = 1000       # 参与训练的问答对个数
        self.input_seq_len = 25         # 输入序列长度
        self.output_seq_len = 50        # 输出序列长度
        self.init_learning_rate = 0.5     # 初始学习率
        self.wordToken = word_token.WordToken()

        # 放在全局的位置，为了动态算出 num_encoder_symbols 和 num_decoder_symbols
        self.max_token_id = self.wordToken.load_file_list(['./dialog/question', './dialog/answer'], self.min_freq)
        self.num_encoder_symbols = self.max_token_id + 5
        self.num_decoder_symbols = self.max_token_id + 5

        self.sess = tf.Session()
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = self.get_model(feed_previous=True)
        saver.restore(self.sess, './model/'+str(self.epochs)+'/demo_')

        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.target_weights = target_weights
        self.outputs = outputs        
        
    def get_id_list_from(self,sentence):
        """
        得到分词后的ID
        """
        sentence_id_list = []
        seg_list = jieba.cut(sentence)
        for str in seg_list:
            id = self.wordToken.word2id(str)
            if id:
                sentence_id_list.append(self.wordToken.word2id(str))
        return sentence_id_list


    def seq_to_encoder(self,input_seq):
        """
        从输入空格分隔的数字id串，转成预测用的encoder、decoder、target_weight等
        """
        input_seq_array = [int(v) for v in input_seq.split()]
        encoder_input = [self.PAD_ID] * (self.input_seq_len - len(input_seq_array)) + input_seq_array
        decoder_input = [self.GO_ID] + [self.PAD_ID] * (self.output_seq_len - 1)
        encoder_inputs = [np.array([v], dtype=np.int32) for v in encoder_input]
        decoder_inputs = [np.array([v], dtype=np.int32) for v in decoder_input]
        target_weights = [np.array([1.0], dtype=np.float)] * self.output_seq_len
        return encoder_inputs, decoder_inputs, target_weights

    def get_model(self,feed_previous=False):
        """
        构造模型
        """
        learning_rate = tf.Variable(float(self.init_learning_rate), trainable=False, dtype=tf.float32)
        learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)

        encoder_inputs = []
        decoder_inputs = []
        target_weights = []
        for i in range(self.input_seq_len):
            encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in range(self.output_seq_len + 1):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
        for i in range(self.output_seq_len):
            target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

        # decoder_inputs左移一个时序作为targets
        targets = [decoder_inputs[i + 1] for i in range(self.output_seq_len)]

        cell = tf.contrib.rnn.BasicLSTMCell(self.size)

        # 这里输出的状态我们不需要
        outputs, _ = seq2seq.embedding_attention_seq2seq(
                            encoder_inputs,
                            decoder_inputs[:self.output_seq_len],
                            cell,
                            num_encoder_symbols=self.num_encoder_symbols,
                            num_decoder_symbols=self.num_decoder_symbols,
                            embedding_size=self.size,
                            output_projection=None,
                            feed_previous=feed_previous,
                            dtype=tf.float32)

        # 计算加权交叉熵损失
        loss = seq2seq.sequence_loss(outputs, targets, target_weights)
        # 梯度下降优化器
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        # 优化目标：让loss最小化
        update = opt.apply_gradients(opt.compute_gradients(loss))
        # 模型持久化
        saver = tf.train.Saver(tf.global_variables())

        return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate
    

    def predict_answer(self,input_seq):
        """
        预测过程    
        """        
        input_seq = input_seq.strip()
        input_id_list = self.get_id_list_from(input_seq)
        if (len(input_id_list)):
            sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = self.seq_to_encoder(' '.join([str(v) for v in input_id_list]))

            input_feed = {}
            for l in range(self.input_seq_len):
                input_feed[self.encoder_inputs[l].name] = sample_encoder_inputs[l]
            for l in range(self.output_seq_len):
                input_feed[self.decoder_inputs[l].name] = sample_decoder_inputs[l]
                input_feed[self.target_weights[l].name] = sample_target_weights[l]
                input_feed[self.decoder_inputs[self.output_seq_len].name] = np.zeros([2], dtype=np.int32)

        # 预测输出
            outputs_seq = self.sess.run(self.outputs, input_feed)
        # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
            outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
        # 如果是结尾符，那么后面的语句就不输出了
            if self.EOS_ID in outputs_seq:
                outputs_seq = outputs_seq[:outputs_seq.index(self.EOS_ID)]
            outputs_seq = [self.wordToken.id2word(v) for v in outputs_seq]
            if outputs_seq is None:
                return 'sorry,i dont know'
            else:
        
                for index in range(len(outputs_seq)):
                    if outputs_seq[index] is None:
                        outputs_seq[index]=''
            
                return " ".join(outputs_seq)
    
        else:
            return("对不起哦，您的这个问题我暂时回答不了") 
                

if __name__ == '__main__':
    anwser = Answer()
    retMsg = anwser.predict_answer('哈哈') #测试模型预测
    print(retMsg)

