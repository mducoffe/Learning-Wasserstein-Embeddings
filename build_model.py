#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:59:48 2018

@author: mducoffe, rflammary, ncourty
"""
from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import Dense, Activation
from keras.layers import Flatten, Reshape
from keras.layers import Conv2D
from keras.layers import Input, Lambda
from keras.callbacks import ModelCheckpoint,EarlyStopping
from dataset import get_data, MNIST, REPO

MODEL='models'
#%%
def euclidean_distance(vects):
    x, y = vects
    return K.sum(K.square(x - y), axis=(1), keepdims=True)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def sparsity_constraint(y_true, y_pred):
    return K.mean(K.sum(K.sqrt(y_pred+ K.epsilon()), axis=(1,2,3)), axis=0)

def kullback_leibler_divergence_(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.sum(y_true * K.log(y_true / y_pred), axis=(1,2,3)), axis=-1)

def build_model(image_shape=(28,28), embedding_size=50):

    s = image_shape[-1]
    feat=Sequential()
    feat.add(Conv2D(20,(3,3),
            activation='relu',padding='same',
            input_shape=(1, s, s), data_format='channels_first'))
    feat.add(Conv2D(5,(5,5),activation='relu',data_format='channels_first', padding='same'))
    feat.add(Flatten())
    feat.add(Dense(100))
    feat.add(Dense(embedding_size))

    inp1=Input(shape=(1,s,s))
    inp2=Input(shape=(1,s,s))

    feat1=feat(inp1)
    feat2=feat(inp2)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([feat1, feat2])

    feat.compile('sgd','mse')
    model=Model([inp1,inp2],distance)
    model.compile('adam','mse')
    
    unfeat=Sequential()
    input_dim = feat.get_output_shape_at(0)[-1]
    unfeat.add(Dense(100, input_shape=(input_dim,), activation='relu'))
    unfeat.add(Dense(5*s*s, activation='relu'))
    unfeat.add(Reshape((5, s,s)))
    unfeat.add(Conv2D(10,(5,5),activation='relu',data_format='channels_first', padding='same'))
    unfeat.add(Conv2D(1,(3,3),activation='linear',data_format='channels_first', padding='same'))
    unfeat.add(Flatten())
    unfeat.add(Activation('softmax')) # samples are probabilities
    unfeat.add(Reshape((1,s,s)))


    uf1=unfeat(feat1)
    uf2=unfeat(feat2)

    unfeat.compile('adam','kullback_leibler_divergence')
    
    model2=Model([inp1,inp2],[distance, uf1,uf2, uf1, uf2])
    model2.compile('adam',['mse', kullback_leibler_divergence_,kullback_leibler_divergence_,
                           sparsity_constraint, sparsity_constraint],
                           loss_weights=[1, 1e1,1e1, 1e-3, 1e-3])
    
    return {'feat':feat, 'emd':model,'unfeat':unfeat,'dwe':model2}


def train_DWE(dataset_name=MNIST, repo=REPO, embedding_size=50, image_shape=(28,28),\
              batch_size=100, epochs=100):
    
    train, valid, test=get_data(dataset_name, repo)
    dict_models=build_model(image_shape, embedding_size)
    
    model = dict_models['dwe']
    
    n_train=len(train[0])
    steps_per_epoch=int(n_train/batch_size)
    earlystop=EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    saveweights=ModelCheckpoint('{}/{}_autoencoder'.format(MODEL,dataset_name), monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    validation_data=([valid[0],valid[1]],[valid[2], valid[0], valid[1], valid[0], valid[1]])
    test_data=([test[0],test[1]],[test[2], test[0], test[1], test[0], test[1]])
    
    def myGenerator():
        #loading data
        while 1:
            for i in range(steps_per_epoch):
                index=range(i*batch_size, (i+1)*batch_size)
                x1,x2,y=(train[0][index], train[1][index], train[2][index])
                yield [x1,x2],[y, x1,x2, x1, x2]
                
    model.fit_generator(myGenerator(),steps_per_epoch,
           epochs,validation_data=validation_data,
           callbacks=[earlystop, saveweights])
    
    model.evaluate(test_data[0], test_data[1])
    
    for key in dict_models:
        dict_models[key].save('{}/{}_{}.hd5'.format(MODEL, dataset_name, key))

#%%    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--dataset_name', type=str, default='cat', help='dataset name')
    parser.add_argument('--repo', type=str, default=REPO, help='repository')
    parser.add_argument('--embedding_size', type=int, default=50, help='embedding size')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    
    
    args = parser.parse_args()                                                                                                                                                                                                                             
    dataset_name=args.dataset_name
    repo=args.repo
    embedding_size=args.embedding_size
    batch_size=args.batch_size
    epochs=args.epochs
    
    train_DWE(dataset_name, repo, embedding_size, batch_size=batch_size, epochs=epochs)