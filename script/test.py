#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:25:44 2019

@author: zhangjunjie
"""
import tensorflow as tf
import numpy as np
from data import load_data

BatchSize = 50
D = np.load('../test/names_onehots.npy', allow_pickle=True).item()
name = D['names']
test_C, test_feature, test_y = load_data('test')
data_size = test_C.shape[0]

sess = tf.Session()
#load moedel
saver = tf.train.import_meta_graph("./weights/model.meta")
saver.restore(sess, './weights/model')
graph = tf.get_default_graph()
#predict
prediction = []
for i in range(0, data_size, BatchSize):
    test_output = sess.run('output/BiasAdd:0',{'input:0': test_feature[i:i + BatchSize],'C:0': test_C[i:i + BatchSize]})
    pred = np.argmax(test_output, axis=1)
    prediction.extend(list(pred))
sess.close()
f = open('output_517021911128.csv', 'w')
f.write('Chemical,Label\n')
for i, v in enumerate(prediction):
    f.write(name[i] + ',%d\n' % v)
print('predict all')
f.close()
