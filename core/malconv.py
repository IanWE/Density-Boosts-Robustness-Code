from keras.layers import Dense, Conv1D, Activation, GlobalMaxPooling1D, Input, Embedding, Multiply
from keras.models import Model
from keras import backend as K
from keras import metrics
import os
import math
import random
import argparse
import os
import numpy as np
import requests


def Malconv():
    def __init__(self):
        batch_size = 100
        input_dim = 257 # every byte plus a special padding symbol
        padding_char = 256
        
        if os.path.exists('ember/malconv/malconv.h5'):
            print("restoring malconv.h5 from disk for continuation training...")
            from keras.models import load_model
            basemodel = load_model('ember/malconv/malconv.h5')
            _, maxlen, embedding_size = basemodel.layers[1].output_shape
            input_dim
        else:
            maxlen = 2**20 # 1MB
            embedding_size = 8 
        
            # define model structurvimapp
            inp = Input( shape=(maxlen,))
            emb = Embedding( input_dim, embedding_size )( inp )
            filt = Conv1D( filters=128, kernel_size=500, strides=500, use_bias=True, activation='relu', padding='valid' )(emb)
            attn = Conv1D( filters=128, kernel_size=500, strides=500, use_bias=True, activation='sigmoid', padding='valid')(emb)
            gated = Multiply()([filt,attn])
            feat = GlobalMaxPooling1D()( gated )
            dense = Dense(128, activation='relu')(feat)
            outp = Dense(1, activation='sigmoid')(dense)
        
            self.base_model = basemodel = Model( inp, outp )
        
        basemodel.summary() 
    def predict(self,X):
        r = self.basemodel.predict(X)[:,0]
	return r

