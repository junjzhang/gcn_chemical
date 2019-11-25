#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:51:53 2019

@author: zhangjunjie
"""

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
import scipy.sparse as sp
from scipy.sparse.base import spmatrix
from sklearn.metrics import accuracy_score, f1_score


def evaluate(preds, labels, masks):
    masked_node_indices = np.nonzero(masks)
    masked_preds = preds[masked_node_indices]
    masked_labels = labels[masked_node_indices]

    accuracy = accuracy_score(masked_labels, masked_preds)
    macro_f1 = f1_score(masked_labels, masked_preds, pos_label=None, average='macro')
    micro_f1 = f1_score(masked_labels, masked_preds, pos_label=None, average='micro')

    return accuracy, macro_f1, micro_f1

class GCNLayer(keras.model):
    def __init__(self, n_units,activation=tf.nn.relu, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.n_units = n_units
        self.activation = activation
        self.W = None
        self.b = None
    
    def build(self, input_shape):
        input_dim = input_shape[1][1]
        self.W = self.add_weight("W", shape = [input_dim,self.n_units])
        self.b = self.add_weight("b", shape = [self.num_units], initializer=tf.zeros_initializer)
        super().build(input_shape)
        
    def call(self, inputs, traininig=None, mask=None):
        A, H = inputs
        A_is_sparse = isinstance(A, tf.SparseTensor)
        H_is_sparse = isinstance(H, tf.SparseTensor)
        
        if H_is_sparse:
            HW = tf.sparse_tensor_dense_matmul(H,self.W) + self.b
        else:
            HW = tf.matmul(H,self.W) + self.b
        
        if A_is_sparse:
            AHW = tf.sparse_tensor_dense_matmul(A,HW)
        else:
            AHW = tf.matmul(A,HW)
            
        if self.activation is not None:
            AHW = self.activation(AHW)
        return AHW
    
    def l2_loss(self):
        return tf.nn.l2_loss(self.W)
    
class GCN(keras.Model):
    def __init__(self, n_units_list, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.n_units_list = n_units_list
        self.gcn_funcs = []
        
        for i,n_units in enumerate(n_units_list):
            activation = tf.nn.relu if i < len (n_units_list) - 1 else None
            gcn_func = GCNLayer(n_units, activation)    
            self.gcn_funcs.append(gcn_func)
            
    def l2_loss(self):
        return tf.add_n([gcn_func.l2_loss() for gcn_func in self.gcn_funcs])
    
    def call(self,inputs, training = None, mask = None):
        A, H = inputs
        for i, gcn_func in enumerate(self.gcn_funcs):
            H = gcn_func([A, H], training=training)
            if i < len(self.gcn_funcs) - 1:
                H = self.dropout_layer(H, training=training)
        return H
    
    @classmethod
    def gcn_kernal(cls, adj):
        inv_D = np.array(adj.sum(axis=1)).flatten()
        inv_D = np.power(inv_D, -0.5)
        inv_D[np.isinf(inv_D)] = 0.0
        inv_D = sp.diags(inv_D)
        return inv_D.dot(adj).dot(inv_D) + np.eye(inv_D.shape[0])
    
    @classmethod
    def gcn_kernal_tensor(cls, adj, sparse=True):
        adj = GCN.gcn_kernal(adj)
        if sparse:
            A = adj.tocoo().astype(np.float32)
            A = tf.SparseTensor(indices=np.stack((A.row, A.col), axis=1), values=A.data, dense_shape=A.shape)
        else:
            A = tf.Variable(adj.todense().astype(np.float32), trainable=False)
        return A
    
class GCNTrainer(object):
    def __init__(self, gcn_model):
        self.model = gcn_model

    def train(self,
              adj,
              feature_matrix,
              labels,
              train_masks,
              test_masks,
              steps=1000,
              learning_rate=1e-3,
              l2_coe=1e-3,
              show_interval=20,
              eval_interval=20):

        if test_masks is None:
            test_masks = 1 - np.array(train_masks)

        A = GCN.gcn_kernal_tensor(adj, sparse=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        if feature_matrix is None:
            feature_matrix = sp.diags(range(adj.shape[0]))

        if isinstance(feature_matrix, spmatrix):
            coo_feature_matrix = feature_matrix.tocoo().astype(np.float32)
            x = tf.SparseTensor(indices=np.stack((coo_feature_matrix.row, coo_feature_matrix.col), axis=1),
                                values=coo_feature_matrix.data, dense_shape=coo_feature_matrix.shape)
        else:
            x = tf.Variable(feature_matrix, trainable=False)

        num_masked = tf.cast(tf.reduce_sum(train_masks), tf.float32)
        for step in range(steps):
            with tf.GradientTape() as tape:
                logits = self.model([A, x], training=True)
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels = labels)
                losses *= train_masks
                mean_loss = tf.reduce_sum(losses) / num_masked
                loss = mean_loss + self.model.l2_loss() * l2_coe

            watched_vars = tape.watched_variables()
            grads = tape.gradient(loss, watched_vars)
            optimizer.apply_gradients(zip(grads, watched_vars))

            if step % show_interval == 0:
                print("step = {}\tloss = {}".format(step, loss))

            if step % eval_interval == 0:
                preds = self.model([A, x])
                preds = tf.argmax(preds, axis=-1).numpy()
                accuracy, macro_f1, micro_f1 = evaluate(preds, labels, test_masks)
                print("step = {}\taccuracy = {}\tmacro_f1 = {}\tmicro_f1 = {}".format(step, accuracy, macro_f1, micro_f1))


        