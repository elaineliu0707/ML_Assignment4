import re
import sys
import shutil as sh
import os

path = "/home/yingjie/Desktop/Machine_Learning/Assignment4/data/"
path2 = "/home/yingjie/Desktop/Machine_Learning/Assignment4/data_clean/"
#path = "/Users/Pan/Google Drive/Data Science/Machine Learning/ML_Assignment4/data/"
#path2 = "/Users/Pan/Google Drive/Data Science/Machine Learning/ML_Assignment4/data_clean/"

def data_preprocess(path,thefile):
    # Replace all numbers with "NUM" special character
    num_dec = re.compile("[\d]+\.[\d]+")
    num_comma = re.compile("[\d]+,[\d]+")
    num_reg = re.compile("[\d]+")

    # Detect non-English characters
    non_eng = re.compile("[^a-zA-Z\n ]")
    
    #open file
    file=open(os.path.join(path,thefile), "r", encoding="iso-8859-15")
    text = file.read().lower()
    file.close()
    
    # Replace various forms of numbers with NUM character
    text = num_dec.sub("NUM", text)
    text = num_comma.sub("NUM", text)
    text = num_reg.sub("NUM", text)

    # Replace all non-English characters with a space
    text = non_eng.sub(" ", text)

    # Replace double spaces with a single space
    text = text.replace("  ", " ")
    
    # Split the file into reviews based on \n
    text = text.split("\n")
        
    return(text)

train=data_preprocess(path,"down_sampled_reviews/train_tiny.txt")
valid=data_preprocess(path,"down_sampled_reviews/valid_tiny.txt")

import numpy as np
from tensorflow.contrib import learn

def textlst_2_vocab(textlst,num_of_reviews=None,seq_size = 2):
    if num_of_reviews is None:
        num_of_reviews=len(textlst)
        
    small_train=''
    for i in range(num_of_reviews):
        small_train+=textlst[i]+" END "
    small_train=[small_train]
    
    max_document_length=len(small_train[0].split(" "))
    
    ## Create the vocabularyprocessor object, setting the max lengh of the documents.
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    
    ## Transform the documents using the vocabulary.
    train_transformed = np.array(list(vocab_processor.fit_transform(small_train)))
    
    ## Extract word:id mapping from the object.
    vocab_dict = vocab_processor.vocabulary_._mapping
    vocab_dict['END'] = -1
    
    train_transformed2=np.array(list(vocab_processor.fit_transform(small_train)))
    train_transformed2=[np.trim_zeros(train_transformed2[0], 'b')]
    
    #sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
    sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
    ## Treat the id's as index into list and create a list of words in the ascending order of id's
    ## word with id i goes at index i of the list.
    vocabulary = list(list(zip(*sorted_vocab))[1])
    
    trainX, trainY = [], []
    max_document_length=len(train_transformed2[0])
    for i in range(max_document_length): 
        #[0,1,2,3,4,5,-1,1,2,3,4,5]
        if -1 not in train_transformed2[0][i:i+seq_size+1]:
            trainX.append(np.expand_dims(train_transformed2[0][i:i+seq_size], axis=1).tolist())
            trainY.append(train_transformed2[0][i+1:i+seq_size+1])
    
    size_vocab = len(vocabulary)
    
    return trainX,trainY,size_vocab

num_of_reviews=50
seq_size=2
trainX,trainY,size_vocab=textlst_2_vocab(train,num_of_reviews=num_of_reviews,seq_size = seq_size)
print(size_vocab)
print("X:\n",'\n'.join([str(item) for item in trainX[0:5]]))
print("Y:\n",'\n'.join([str(item) for item in trainY[0:5]]))

num_of_reviews_valid = 10
validX,validY,size_vocab_val=textlst_2_vocab(valid,num_of_reviews=num_of_reviews_valid,seq_size = seq_size)
print(size_vocab_val)
print("X:\n",'\n'.join([str(item) for item in validX[0:5]]))
print("Y:\n",'\n'.join([str(item) for item in validY[0:5]]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.preprocessing import MinMaxScaler
import timeit

# Build computational graph
input_dim=1 # dim > 1 for multivariate time series
hidden_dim=100 # number of hiddent units h
graph = tf.Graph()

with graph.as_default():
    # input place holders
    # input Shape: [# training examples, sequence length, # features]
    x = tf.placeholder(tf.float32,[None,seq_size,input_dim])
    print("input shape:", x.shape)
    
    # label Shape: [# training examples, sequence length]
    y = tf.placeholder(tf.int32,[None,seq_size])
    print("ground truth shape:", y.shape)
    
    num_examples = tf.shape(x)[0]
    
    # RNN Network
    cell = rnn.BasicRNNCell(hidden_dim)
    
    # RNN output Shape: [# training examples, sequence length, # hidden] 
    outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    print("output from tf,nn.dynamic_rnn shape", outputs.shape)
    
    #outputs = tf.reshape(outputs, [num_examples*seq_size, hidden_dim])
    print("reshaped output from RNN shape:", outputs.shape)
    
    # weights for output dense layer (i.e., after RNN)
    # W shape: [# hidden, 1]
    W_out = tf.Variable(tf.random_normal([hidden_dim,size_vocab]),name="w_out")
    print("weights shape:", W_out.shape)
    # b shape: [1]
    b_out = tf.Variable(tf.random_normal([size_vocab]),name="b_out")
    print("bias shape",b_out.shape)
    
    # logit shape [# training examples, vocab_size] 
    outputs_reshaped = tf.reshape(outputs, [num_examples*seq_size, hidden_dim])
    print("fully connected and reshaped outputs:", outputs_reshaped) 
    
    network_output = ((tf.matmul(outputs_reshaped,W_out) + b_out))
    #print(network_output)
          
    y_pred = tf.reshape(network_output,[num_examples, seq_size, size_vocab])
    print("final logits shape:", y_pred.shape)
    
    # Cost & Training Step
    #cost = -tf.reduce_sum(y*tf.log(y_pred)) #cross_entropy
    weights = tf.ones([num_examples,seq_size])
    loss = tf.contrib.seq2seq.sequence_loss(y_pred,y,weights) 
    #cost = tf.reduce_sum(loss)/num_examples
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

max_itr=2
batch_size=32
batch_num=len(trainX)//batch_size
print("batch_num: " +str(batch_num))

# Run Session

with tf.Session(graph=graph) as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())
    # Run for 1000 iterations (1000 is arbitrary, need a validation set to tune!)
    print('Training...')
    start=timeit.default_timer()
    print("start time: %10.5f"%(start))
    
    for i in range(max_itr): 
        print("iter %3d" %(i))  
        #print ("max batch num: "+str(batch_num))
        
        saver_train_error = 0
        for NUM in range(1,batch_num):
            batch_trainX=trainX[(NUM-1)*batch_size:NUM*batch_size]
            batch_trainY=trainY[(NUM-1)*batch_size:NUM*batch_size]
            
            _, train_err = sess.run([train_op,loss],feed_dict={x:batch_trainX,y:batch_trainY})

            if  NUM % 100 == 0: 
                print("  ##batch NUM %6d :" %(NUM))
                print('    step, train err= %6d: %8.5f' % (i+1,train_err))
            saver_train_error += train_err
            mean_train_error = saver_train_error/NUM
            #print('  step, test err= %6d: %8.5f' % (i+1,test_err))
            
    #print("  ##batch NUM %6d :" %(NUM))
        print('    step, mean train err= %6d: %8.5f' % (i+1,mean_train_error))
    
    ##### Test trained model on training data
    #predicted_vals_all= sess.run(y_pred,feed_dict={x:trainX})
    
    # Get predicted value in each predicted sequence:
    #predicted_vals = np.argmax(predicted_vals_all,2) 

    #print("Dimension check for train:"+str(np.size(predicted_vals_all,0)==len(trainX)))
    #myaccuracy=sum([el[1] for el in predicted_vals==trainY])/np.size(predicted_vals_all,0)
    #print('my training accuracy = %8.5f' % myaccuracy)
    
    #end=timeit.default_timer()
    #print("Training time : %10.5f"%(end-start))
    ####### Test trained model on testing data
        
    test_err = sess.run(loss,feed_dict={x:validX,y:validY})
    print('test err=  %8.5f' % (test_err))
    print('---------------------------------')
    # testing error:
    #predicted_vals_all_test= sess.run(y_pred,feed_dict={x:validX})
    
    # Get predicted value in each predicted sequence:
    #predicted_vals_test = np.argmax(predicted_vals_all_test,2) 
    
    # testing accuracy:
    #myaccuracy_test=sum([el[1] for el in predicted_vals_test==validY])/np.size(predicted_vals_all_test,0)
    #print('my testing accuracy = %8.5f' % myaccuracy_test)
    #print('---------------------------------')
import timeit
# Encapsulate the entire prediction problem as a function
def build_and_predict(trainX,trainY,validX,cell,cellType,input_dim=1,hidden_dim=100,seq_size = 2,max_itr=200):

    graph = tf.Graph()
    with graph.as_default():
        # input place holders
        # input Shape: [# training examples, sequence length, # features]
        x = tf.placeholder(tf.float32,[None,seq_size,input_dim])
        print("input shape:", x.shape)
    
        # label Shape: [# training examples, sequence length]
        y = tf.placeholder(tf.int32,[None,seq_size])
        print("ground truth shape:", y.shape)
    
        num_examples = tf.shape(x)[0]
    
        # RNN Network
        cell = rnn.BasicRNNCell(hidden_dim)
    
        # RNN output Shape: [# training examples, sequence length, # hidden] 
        outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
        print("output from tf,nn.dynamic_rnn shape", outputs.shape)
    
        #outputs = tf.reshape(outputs, [num_examples*seq_size, hidden_dim])
        print("reshaped output from RNN shape:", outputs.shape)
    
        # weights for output dense layer (i.e., after RNN)
        # W shape: [# hidden, 1]
        W_out = tf.Variable(tf.random_normal([hidden_dim,size_vocab]),name="w_out")
        print("weights shape:", W_out.shape)
        # b shape: [1]
        b_out = tf.Variable(tf.random_normal([size_vocab]),name="b_out")
        print("bias shape",b_out.shape)
    
        # logit shape [# training examples, vocab_size] 
        outputs_reshaped = tf.reshape(outputs, [num_examples*seq_size, hidden_dim])
        print("fully connected and reshaped outputs:", outputs_reshaped) 
    
        network_output = ((tf.matmul(outputs_reshaped,W_out) + b_out))
        #print(network_output)
          
        y_pred = tf.reshape(
            network_output,
            [num_examples, seq_size, size_vocab])
        print("final logits shape:", y_pred.shape)
    
        # Cost & Training Step
        #cost = -tf.reduce_sum(y*tf.log(y_pred)) #cross_entropy
        weights = tf.ones([num_examples,seq_size])
        loss = tf.contrib.seq2seq.sequence_loss(y_pred,y,weights) 
        #cost = tf.reduce_sum(loss)/num_examples
        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        
        # Run Session
    with tf.Session(graph=graph) as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # Run for 1000 iterations (1000 is arbitrary, need a validation set to tune!)
        start=timeit.default_timer()
        print('Training %s ...'%cellType)
        for i in range(max_itr): 
            print("iter %3d" %(i))  
        #print ("max batch num: "+str(batch_num))
        
            saver_train_error = 0
            for NUM in range(1,batch_num):
                batch_trainX=trainX[(NUM-1)*batch_size:NUM*batch_size]
                batch_trainY=trainY[(NUM-1)*batch_size:NUM*batch_size]
            
                _, train_err = sess.run([train_op,loss],feed_dict={x:batch_trainX,y:batch_trainY})

                if  NUM % 100 == 0: 
                    print("  ##batch NUM %6d :" %(NUM))
                    print('    step, train err= %6d: %8.5f' % (i+1,train_err))
                saver_train_error += train_err
                mean_train_error = saver_train_error/NUM
            #print('  step, test err= %6d: %8.5f' % (i+1,test_err))
            
    #print("  ##batch NUM %6d :" %(NUM))
        print('    step, mean train err= %6d: %8.5f' % (i+1,mean_train_error))
        end=timeit.default_timer()        
        print("Training time : %10.5f"%(end-start))
         # Test trained model on training data
        predicted_vals_all= sess.run(y_pred,feed_dict={x:validX}) 
        # Get last item in each predicted sequence:
        #predicted_vals = predicted_vals_all[:,seq_size-1]
      
    
    return predicted_vals_all

input_dim=1 # dim > 1 for multivariate time series
hidden_dim=100 # number of hiddent units h
max_itr=2 # number of training iterations

# Different RNN Cell Types
RNNcell = rnn.BasicRNNCell(hidden_dim)
LSTMcell = rnn.BasicLSTMCell(hidden_dim)
GRUcell = rnn.GRUCell(hidden_dim)

# Build models and predict on testing data
predicted_vals_rnn=build_and_predict(trainX,trainY,validX,RNNcell,"RNN",input_dim,hidden_dim,seq_size,max_itr)
print('RNN END---------------------------------')
predicted_vals_lstm=build_and_predict(trainX,trainY,validX,LSTMcell,"LSTM",input_dim,hidden_dim,seq_size,max_itr)
print('LSTM END---------------------------------')
predicted_vals_gru=build_and_predict(trainX,trainY,validX,GRUcell,"GPU",input_dim,hidden_dim,seq_size,max_itr)
print('GPU END---------------------------------')
