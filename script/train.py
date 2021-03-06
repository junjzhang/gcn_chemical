#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:21:47 2019

@author: zhangjunjie
"""

import numpy as np
import tensorflow as tf
from data import load_data
from gcn import net


LR = 0.001
BatchSize = 50
EPOCH = 100

train_C, train_feature, train_y = load_data('train')
valid_C, valid_feature,valid_y= load_data('validation')
train_size = train_feature.shape[0]

init_op, train_op, loss,accuracy, f1_score, precision, recall = net(train_feature.shape[1:])
sess = tf.Session()
sess.run(init_op)

for epoch in range(EPOCH):
    #shuffle
    shuffled_indices=np.random.permutation(train_C.shape[0])
    train_feature = train_feature[shuffled_indices]
    train_y = train_y[shuffled_indices]
    train_C = train_C[shuffled_indices]
    #banlanced feed
    nonzero_index = np.nonzero(train_y)[0]
    zero_index = np.delete(np.array(range(train_size)),nonzero_index)
    size = 17+int(epoch/60)
    #learning rate decay
    lr = LR*pow(0.9,int(epoch/30))
    for i in range(0, nonzero_index.shape[0], size):
        index = np.hstack((nonzero_index[i:i+size],np.random.choice(zero_index,50-size)))
        b_x, b_y, b_C = train_feature[index], train_y[index], train_C[index]
        _, loss_= sess.run([train_op, loss], {'input:0': b_x, 'label:0': b_y, 'C:0':b_C, 'LR:0': lr})
    if epoch % 5 == 0:
        accuracy_ = 0
        f1_ = 0
        r_ = 0
        for i in range(0, valid_feature.shape[0], BatchSize):
            b_x, b_y, b_C = valid_feature[i:i + BatchSize], valid_y[i:i + BatchSize], valid_C[i:i + BatchSize]
            accu, f1u, r= sess.run([accuracy, f1_score, recall], {'input:0': b_x, 'label:0': b_y, 'C:0':b_C})
            accuracy_ += accu
            f1_ += f1u
            r_ += r
        accuracy_ = accuracy_ * BatchSize / valid_feature.shape[0]
        f1_ = f1_ * BatchSize / valid_feature.shape[0]
        r_ = r * BatchSize / valid_feature.shape[0]
        print('epoch:', epoch, '| train loss: %.3f' % loss_, '| valid f1: %.4f' % f1_, '| valid acc: %.2f' % accuracy_, '| valid recall: %.4f' % r_)
#save    
saver = tf.train.Saver()
saver.save(sess, './weights/model')
sess.close()

