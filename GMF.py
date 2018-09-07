'''
Created on Aug 9, 2016

Keras Implementation of Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, concatenate
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
import pickle

import os # add by Aodong
from hyperparams import Hyperparams as hp #yueqiu

import pickle
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')

    # added by Aodong
    parser.add_argument('--chunk_id', type=int, default=0,
                        help='Data chunk id.')
    parser.add_argument('--mini', type=int, default=0,
                    help='Whether to use the mini data.') #yueqiu
    parser.add_argument('--gt', type=int, default=0,
                        help='include confidence or not')  #yueqiu

    return parser.parse_args()

#def init_normal():
#    return initializers.RandomNormal(stddev=0.01)

def get_model(num_users, num_items, latent_dim, regs=[0,0], if_cat = 0):
    num_cat = 18
    
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    if if_cat:
        cat_input = Input(shape=(num_cat,), dtype='float', name = 'cat_input')
 #   MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
 #                                 init = init_normal, W_regularizer = l2(regs[0]), input_length=1)
 #   MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
 #                                 init = init_normal, W_regularizer = l2(regs[1]), input_length=1)   

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01), W_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01), W_regularizer = l2(regs[1]), input_length=1) 
    #MF_Embedding_Cat = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'category_embedding',
    #                              init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01), W_regularizer = l2(regs[1]), input_length=num_cat)     
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    if if_cat:
        item_cat_merge = concatenate([item_latent, cat_input])
        item_cat_latent = Dense(latent_dim, activation=None, init='lecun_uniform', name = 'item_cat_latent')(item_cat_merge)
        predict_vector = merge([user_latent, item_cat_latent], mode = 'mul')
    else:
        predict_vector = merge([user_latent, item_latent], mode = 'mul')
    # Element-wise product of user and item embeddings 
    
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(predict_vector)
    
    if if_cat:
        model = Model(inputs=[user_input, item_input, cat_input], 
                    outputs=prediction)
    else:
        model = Model(inputs=[user_input, item_input], 
                    outputs=prediction)
    return model

def get_train_instances(train, num_negatives, prepath):
    user_input, item_input, labels = [],[],[]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose

    mini = args.mini #yueqiu
    chunk_id = '_' + str(args.chunk_id) # added by Aodong
    gt = args.gt   #yueqiu
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("GMF arguments: %s" %(args))
    sys.stdout.flush()
    prepath_out = hp.prepath_out + hp.fn if gt == 1 else hp.prepath_out #yueqiu
    prepath_out = prepath_out + 'mini_' if mini == 1 else prepath_out #yueqiu
    model_out_file = prepath_out + 'GMF_%d.h5' %(num_factors) # modified by Aodong
    
    # Loading data
    t1 = time()
    prepath = hp.prepath + hp.fn if gt == 1 else hp.prepath #yueqiu
    prepath = prepath + 'mini_' if mini == 1 else prepath #yueqiu
    train, testRatings, testNegatives = pickle.load(open(prepath + "mat" + chunk_id + ".p", "rb" )), pickle.load(open(prepath + "testRatings" + chunk_id + ".p","rb")), pickle.load(open(prepath + "testNegatives" + chunk_id + ".p","rb")) # modified by Aodong
    num_users, num_items = pickle.load(open(prepath + "num_users" + chunk_id + ".p", "rb" )), pickle.load(open(prepath + "num_items" + chunk_id + ".p", "rb" )) # modified by Aodong
    #print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
    #      %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, num_factors, regs)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=op(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    #print(model.summary())
    
    # Load model, added by Aodong
    if os.path.exists(model_out_file): 
        model.load_weights(model_out_file)

    # Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    #mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
    #p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
    sys.stdout.flush()
    
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if_cat = False
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances

        user_input, item_input, labels = get_train_instances(train, num_negatives, prepath)
        
        # Training
        if if_cat:
            hist = model.fit([np.array(user_input), np.array(item_input), np.array(cat_input)], #input
                             np.array(labels), # labels 
                             batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        else:
            hist = model.fit([np.array(user_input), np.array(item_input)], #input
                             np.array(labels), # labels 
                             batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)

        t2 = time()
        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            sys.stdout.flush()
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best GMF model is saved to %s" %(model_out_file))
    sys.stdout.flush()