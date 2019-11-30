#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:28:27 2019

@author: zhangjunjie
"""
import tensorflow as tf

def net(input_shape):
    if not isinstance(input_shape, list):
        input_shape = list(input_shape)
    input = tf.placeholder(tf.float32, [None] + input_shape, name='input')
    C = tf.placeholder(tf.float32,[None,input_shape[0],input_shape[0]], name='C')
    LR = tf.Variable(0.001, name = "LR")
    label = tf.placeholder(tf.int32, [None], name='label')
    label = tf.one_hot(label, 2)
    #Conv layer one
    w1 = tf.Variable(tf.random_normal([input_shape[1], 60], stddev=pow(2/(input_shape[1]+60),0.5)), name='w1')
    b1 = tf.Variable(tf.random_normal([input_shape[0], 60]), name='b1')
    h1 = tf.add(tf.tensordot(input,w1,axes=1),b1)
    h1 = tf.matmul(C,h1)
    h1 = tf.nn.relu(h1)
    #Conv layer two
    w2 = tf.Variable(tf.random_normal([60, 45], stddev=pow(2/(60+45),0.5)), name='w2')
    #tf.nn.dropout(w2,keep_prob = 0.999)
    b2 = tf.Variable(tf.random_normal([input_shape[0],45],stddev=1), name='b2')
    h2 = tf.add(tf.tensordot(h1,w2,axes=1),b2)
    h2 = tf.matmul(C,h2)
    h2 = tf.nn.relu(h2)
    #Conv layer three
    w3 = tf.Variable(tf.random_normal([45, 35], stddev=pow(2/(35+45),0.5)), name='w3')
    #tf.nn.dropout(w3,keep_prob = 0.999)
    b3 = tf.Variable(tf.random_normal([input_shape[0],35],stddev=1), name='b3')
    h3 = tf.add(tf.tensordot(h2,w3,axes=1),b3)
    h3 = tf.matmul(C,h3)
    h3 = tf.nn.relu(h3)
    #Conv layer four
    w4 = tf.Variable(tf.random_normal([35, 30], stddev=pow(2/(35+30),0.5)), name='w4')
    #tf.nn.dropout(w2,keep_prob = 0.999)
    b4 = tf.Variable(tf.random_normal([input_shape[0],30],stddev=1), name='b4')
    h4 = tf.add(tf.tensordot(h3,w4,axes=1),b4)
    h4 = tf.matmul(C,h4)
    h4 = tf.nn.relu(h4)
    flat = tf.reduce_sum(h4,axis = 1)
    #full connected
    h5 = tf.keras.layers.Dense(30,activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.0001))(flat)
    #full connected
    h6 = tf.keras.layers.Dense(30,activation=tf.tanh,kernel_regularizer=tf.keras.regularizers.l2(0.0001))(h5)

    output = tf.keras.layers.Dense(2, name='output',activation='sigmoid')(h6)
    class_weights = tf.constant([[1.0, 1.0]])
    # deduce weights for batch samples based on their true label
    weights = tf.reduce_sum(class_weights * label, axis=1)
    #loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=output, weights = weights)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    #metrics
    accuracy = tf.metrics.accuracy(labels=tf.argmax(label, axis=1), predictions=tf.argmax(output, axis=1))[1]
    precision = tf.metrics.precision(labels=tf.argmax(label, axis=1), predictions=tf.argmax(output, axis=1))[1]
    recall = tf.metrics.recall(labels=tf.argmax(label, axis=1), predictions=tf.argmax(output, axis=1))[1]
    f1_score = (2 * precision * recall) / (precision + recall)
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    return init_op, train_op, loss,accuracy, f1_score, precision, recall
