#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:57:29 2019

@author: zhangjunjie
"""
import numpy as np
import networkx as nx
from rdkit import Chem
from pysmiles import read_smiles
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
s=132

def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   element=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   hcount=atom.GetNumImplicitHs(),
                   aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   order=bond.GetBondType())
    return G

def sm2graph(smiles,size, weight = None):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = mol_to_nx(mol)
    except:
        mol = read_smiles(smiles)
    #normalized Laplacian matrix
#    nL = nx.normalized_laplacian_matrix(mol,weight = weight).todense().A
#    nL = np.pad(nL,(0,size-nL.shape[0]))
    #adjacent matrix
    adj = nx.to_numpy_matrix(mol, weight=weight).A
    adj = adj+np.eye(adj.shape[0])
    adj = np.pad(adj,(0,size-adj.shape[0]))
    #degree matrix
    de = np.zeros((size,size))
    for i in mol.degree:
        de[i[0]-1][i[0]-1] = i[1]+1
    #feature
    mole = mol.nodes(data='element')
    #random walk normalized Laplacian matrix
    di = de
    di[di!=0] = 1/di[di!=0]
    rwL = di@adj
    #is aromatic
    ar = mol.nodes(data = 'aromatic')
    return rwL, mole, ar

def process_data(filefolder,weight=None):
    periodic_table = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
                  'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 
                  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
                  'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr',
                  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue')
    filefolder = '../'+filefolder
    data_path = filefolder + '/names_smiles.csv'
    data_rwL = []
#    data_nL = []
    feature = []
    with open(data_path,'r') as f:
        header = f.readline().replace('\n','').split(',')
        if header[0] == 'SMILES':
            index = 0
        else:
            index = 1
        for line in f.readlines():
            line = line.replace('\n','').split(',')
            smiles = str(line[index])
            rwL,mole, ar= sm2graph(smiles,s,weight)
            data_rwL.append(rwL)
#            data_nL.append(nL)
            onehot = np.zeros((s,83))
            i=0
            for atom in mole:
                if (ar[i]):
                    onehot[i][82] = 1
                if(isinstance(atom[1],int)):
                    for j in range(82):
                        if (atom[1] == j+1):
                            onehot[i][j] = 1
                            break
                else:
                    for j in range(82):
                        if (atom[1] == periodic_table[j]):
                            onehot[i][j] = 1
                            break
                i = i+1
            feature.append(onehot)
                        
    np.save(filefolder+'/rwL_matrix.npy',np.array(data_rwL))
#    np.save(filefolder+'/nL_matrix.npy',np.array(data_nL))
    np.save(filefolder+'/feature.npy',np.array(feature))   

def load_data(filefolder):
    ori_filefolder = filefolder
    filefolder = '../'+filefolder
    rwL_filename = filefolder + '/rwL_matrix.npy'
    if (os.path.exists(rwL_filename)==False):
        process_data(ori_filefolder)
    feature_filename = filefolder + '/feature.npy'
    rwL = np.load(rwL_filename)
    feature = np.load(feature_filename)
    if ori_filefolder == 'test':
        label_filename = filefolder + '/output_sample.csv'
    else:
        label_filename = filefolder + '/names_labels.csv'
    label = []
    with open(label_filename,'r') as f:
        header = f.readline().replace('\n','').split(',')
        if header[0]=='Label':
            label_index = 0
        else:label_index = 1
        for line in f.readlines():
            line = line.replace('\n','').split(',')
            label.append(int(line[label_index]))
    label = np.array(label)
    return rwL, feature ,label
