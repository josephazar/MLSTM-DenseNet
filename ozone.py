from __future__ import print_function 
import keras
from keras.layers import Input, Dense, Dropout, Activation, Concatenate,concatenate,multiply, BatchNormalization ,Masking,Reshape,Permute
from keras.models import Model
from keras.layers import Conv1D, GlobalAveragePooling1D, AveragePooling1D
from sklearn.model_selection import StratifiedShuffleSplit
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import pandas as pd 
from utils.utils import transform_labels
import matplotlib.pyplot as plt 
import sklearn 
from utils.layer_utils import AttentionLSTM
seed=7

def read_dataset(dataset_name,t="orig",acr="",inshape=[],rescale=False):
    datasets_dict = {}
    file_name = dataset_name +'/'
    if t=="orig":
        try:
            x_train = np.load(file_name + 'x_train.npy')
            y_train = np.load(file_name + 'y_train.npy')
            x_test = np.load(file_name + 'x_test.npy')
            y_test = np.load(file_name + 'y_test.npy')
        except:
            x_train = np.load(file_name + 'X_train.npy')
            y_train = np.load(file_name + 'y_train.npy')
            x_test = np.load(file_name + 'X_test.npy')
            y_test = np.load(file_name + 'y_test.npy')
            x_train=x_train.reshape(x_train.shape[0],x_train.shape[2],x_train.shape[1])
            x_test=x_test.reshape(x_test.shape[0],x_test.shape[2],x_test.shape[1])
            y_train=np.squeeze(y_train)
            y_test=np.squeeze(y_test)
            
            
        if rescale==True:
            x_train/=1E3
            x_test/=1E3
    
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    return datasets_dict


def standardize(train, test):
     mean=np.mean(train, axis=0)[None,:,:]
     std=np.std(train, axis=0)[None,:,:]
     #X_train_mean = train.mean()
     #X_train_std = train.std()
     #X_train = (train - X_train_mean) / (X_train_std + 1e-8)     
     #X_test = (test - X_train_mean) / (X_train_std + 1e-8)
     # Standardize train and test
     X_train = (train - mean) / std
     X_test = (test - mean) / std
     return X_train, X_test

#reference: https://github.com/houshd/MLSTM-FCN 
def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

#densenet inspired from https://github.com/seasonyc/densenet/blob/master/densenet.py
#not all parameters are used in this experiments   
def DenseNet(input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40,avg_pooling=True):
    """
    Creating a DenseNet
    
    Arguments:
        input_shape  : shape of the input images. E.g. (28,28,1) for MNIST    
        dense_blocks : amount of dense blocks that will be created (default: 3)    
        dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                       or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                       by the given depth (default: -1)
        growth_rate  : number of filters to add per dense block (default: 12)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                       In the paper the authors recommend a dropout of 0.2 (default: None)
        bottleneck   : (True / False) if true it will be added in convolution block (default: False)
        compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                       of 0.5 (default: 1.0 - will have no compression effect)
        weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
        depth        : number or layers (default: 40)
        
    Returns:
        Model        : A Keras model instance
    """
    
    if nb_classes==None:
        raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')
    
    if compression <=0.0 or compression > 1.0:
        raise Exception('Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')
    
    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1))/dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1))//dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
        
        
    
    data_input = Input(shape=input_shape)
    nb_channels = growth_rate * 2
    
    print('Creating DenseNet')
    print('#############################################')
    print('Dense blocks: %s' % dense_blocks)
    print('Layers per dense block: %s' % dense_layers)
    print('#############################################')
          
    y = Masking()(data_input)
    y = AttentionLSTM(8)(y)
    y = Dropout(0.8)(y)
    
    x = Permute((2, 1))(data_input)
    # Initial convolution layer
    x = Conv1D(nb_channels, (5,), padding='same',strides=(1,), use_bias=False, kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    #x = Conv1D(nb_channels, (8,), padding='same',strides=(1,), kernel_initializer='he_uniform')(x)

    

    # Building dense blocks
    for block in range(dense_blocks):
        
        # Add dense block
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck, weight_decay)
        
        if block < dense_blocks - 1:  # if it's not the last dense block
            # Add transition_block
            x = transition_layer(x, nb_channels, dropout_rate, compression, weight_decay,avg_pooling)
            nb_channels = int(nb_channels * compression)
    
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    
    
    
    y = concatenate([y, x])
    y = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(y)
    
    model_name = None
    if growth_rate >= 36:
        model_name = 'widedense'
    else:
        model_name = 'dense'
        
    if bottleneck:
        model_name = model_name + 'b'
        
    if compression < 1.0:
        model_name = model_name + 'c'
        
    return Model(data_input, y, name=model_name), model_name


def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a dense block and concatenates inputs
    """
    
    x_list = [x]
    for i in range(nb_layers):
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)
        x_list.append(cb)
        #x = Concatenate(axis=-1)(x_list)
        #x = Concatenate(axis=-1)([x, cb])
        nb_channels += growth_rate
    x=x_list[0]
    for i in range(1,len(x_list)):
        x = Concatenate(axis=-1)([x, x_list[i]])
    return x, nb_channels


def convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a convolution block consisting of BN-ReLU-Conv.
    """
    # Standard (BN-ReLU-Conv)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x=squeeze_excite_block(x)
    x = Conv1D(nb_channels, (3, ), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),kernel_initializer='he_uniform')(x)
    #x = Conv1D(nb_channels, (3, ), padding='same',kernel_initializer='he_uniform')(x)
   
    # Dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    return x


def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4,avg_pooling=True):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.
    """
    
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(nb_channels*compression), (1, ), padding='same',use_bias=False, kernel_regularizer=l2(weight_decay),kernel_initializer='he_uniform')(x)
    #x = Conv1D(int(nb_channels*compression), (1, ), padding='same',kernel_initializer='he_uniform')(x)
    x=squeeze_excite_block(x)
    # Adding dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    if avg_pooling:
        x = AveragePooling1D((2, ), strides=(2, ))(x)
    return x


def preprocess(X_train,X_test,y_train,y_test,fac,std=True):
    if std:
        X_train,X_test=standardize(X_train,X_test)
    nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))
    
    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) *
                           np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]
    

    # make the min to zero of labels
    y_train,y_test = transform_labels(y_train,y_test)
    
    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64) 
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))
    y_train = enc.transform(y_train.reshape(-1,1)).toarray()
    y_test = enc.transform(y_test.reshape(-1,1)).toarray()
    
    
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[2]//fac,X_train.shape[1]*fac))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[2]//fac,X_test.shape[1]*fac))
    
    return X_train,X_test,y_train,y_test,y_true,nb_classes,class_weight

def train_eval(X_train,X_test,y_train,y_test,nb_classes,class_weight,batch_size=16,epochs=200):
    input_shape = X_train.shape[1:]
    
    classifier =DenseNet(input_shape=input_shape, nb_classes=nb_classes,dense_blocks=2,dropout_rate=None, dense_layers=[4,4],growth_rate=16,avg_pooling=True,compression=0.9)
    model=classifier[0]
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7937, patience=50, min_lr=0.0001,cooldown=0,mode='auto')
    checkpoint = keras.callbacks.ModelCheckpoint("logs/mlstm_densenet_best_model.h5",
                            monitor='loss',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='auto')
    model.fit(X_train, y_train,class_weight=class_weight, batch_size=batch_size, epochs=epochs,shuffle=True, validation_data=(X_test,y_test), callbacks=[reduce_lr,checkpoint])
    
    model.load_weights("logs/mlstm_densenet_best_model.h5")
    scores = model.evaluate(X_test,y_test, verbose=0)
    return scores[1]


def evaluate(X_test,y_test,nb_classes,modelpath):
    input_shape = X_test.shape[1:]
    classifier =DenseNet(input_shape=input_shape, nb_classes=nb_classes,dense_blocks=2,dropout_rate=None, dense_layers=[4,4],growth_rate=16,avg_pooling=True,compression=0.9)
    model=classifier[0]
    model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
    model.load_weights("weights/" + modelpath)
    scores = model.evaluate(X_test,y_test, verbose=0)
    return scores[1]  



name="ozone"
fac=1 #not used 
data=read_dataset(name,"orig",name,[],rescale=False)

X_train=data[name][0]
y_train=data[name][1]
X_test=data[name][2]
y_test=data[name][3]


cv_scores = []

"""
for i in range(10):
    X_tr,X_tst,y_tr,y_tst,y_true,nb_classes,class_weight=preprocess(X_train,X_test,y_train,y_test,fac,std=False)
    score=train_eval(X_tr,X_tst,y_tr,y_tst,nb_classes,class_weight,batch_size=16,epochs=2000)
    cv_scores.append(score*100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
"""

#X=np.concatenate((X_train,X_test))
#y=np.concatenate((y_train,y_test))

"""
#Cross validation
kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=seed)
for train, test in kfold.split(X, y):
    X_train,X_test,y_train,y_test,y_true,nb_classes=preprocess(X[train],X[test],y[train],y[test],fac)
    score=train_eval(X_train,X_test,y_train,y_test,nb_classes,batch_size=16,epochs=150)
    cv_scores.append(score*100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
"""

X_tr,X_tst,y_tr,y_tst,y_true,nb_classes,class_weight=preprocess(X_train,X_test,y_train,y_test,fac,std=False)
score=evaluate(X_tst,y_tst,nb_classes,"ozone.h5")
print("%.2f%% (+/- %.2f%%)" % (score, 0.0))

   
