import sys, os, math
import argparse
from datetime import datetime, date
from mdn import *
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tensorflow import keras
from tensorflow.keras import backend, regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import json
from utils import parse_multiple_json
from PIL import Image, ImageDraw

def loadData1(filePath):
    ## read rehearsal data:
    df=pd.read_csv(filePath, sep=',',header=0)
    df['minDist']=df['minDist']/200
    df['maxDist']=df['maxDist']/200
    df['foodClosest']=df['foodClosest']/15
    df['nestClosest']=df['nestClosest']/15
    df['foodFarthest']=df['foodFarthest']/15
    df['nestFarthest']=df['nestFarthest']/15
    df['foodDist']=df['foodDist']/45
    df['nestDist']=df['nestDist']/45
    data = df.values
    x_input = data[:,0:6]
    y_input = data[:,6:8]
    print(x_input.shape, y_input.shape)
    return x_input, y_input

def generate2(ob, IMG_DIM=24):
    img = Image.new('L', (2*IMG_DIM,IMG_DIM), color=255)
    idraw = ImageDraw.Draw(img)
    lstPts = ob['msgTbl']
    if lstPts is None:
        lstPts = []
    rScale = 0.16/4*IMG_DIM
    for i in range(len(lstPts)):
        x0 = (IMG_DIM/2-1)*(1 + lstPts[i]['dist']/200*math.sin(math.radians(90)-lstPts[i]['angle']))
        y0 = IMG_DIM/2*(1 - lstPts[i]['dist']/200*math.sin(lstPts[i]['angle']))
        c0 = int(lstPts[i]['fHop']*200/15)
        c1 = int(lstPts[i]['nHop']*200/15)
        #print(x0,y0,c0,c1)
        xleft = round(x0-rScale/2+1)
        xright = round(xleft+rScale)-1
        ytop = round(y0-rScale/2)
        ybottom = round(ytop+rScale)-1
        idraw.rectangle([xleft,         ytop, xright,         ybottom], fill=c0, width=1)
        idraw.rectangle([xleft+IMG_DIM, ytop, xright+IMG_DIM, ybottom], fill=c1, width=1)
        #print(x0,y0,c0,c1,' => (',xleft,ytop,'),(',xright,ybottom,')')
    arr = np.array(list(img.tobytes()))
    #print(arr.shape, type(arr))
    arr = np.reshape(arr.astype(np.uint8), (IMG_DIM,2*IMG_DIM,1))
    arr = np.moveaxis(arr, 2, 0)
    #print(arr.shape, type(arr))
    return arr

def loadData2(imgJsonPath, maxLine=None):
    jsonArr, offset = parse_multiple_json(imgJsonPath, maxLoadLine=maxLine)
    print('finish load mgTbl json with total size=', len(jsonArr), ' offset=', offset)
    img_input = []
    for idx in range(len(jsonArr)):
        #jsonMsgTbl = json.loads(jsonArr[idx])
        jsonMsgTbl = jsonArr[idx]
        if idx == 0:
            print('first msgTbl, idx={}, len={}'.format(idx, len(jsonMsgTbl['msgTbl'])))
        img_input.append(generate2(jsonMsgTbl))
    img_input = np.asarray(img_input)
    print(img_input.shape)
    return img_input

#N_HIDDEN = 256
#N_MIXES = 32
#OUTPUT_DIMS = 2
#batch_size = 512
#num_epochs = 500
#loss_type = 'mse'
#loss_type = 'logloss'

def train1(x_input, y_input, args):
    N_HIDDEN = args.n_hidden
    N_MIXES = args.n_mixes
    OUTPUT_DIMS = args.output_dims
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    strCurDateTime = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_file_name = 'mdn_{}_{}_{}_{}_{}'.format(N_HIDDEN, N_MIXES, batch_size, num_epochs, strCurDateTime)
    #begin to build the model
    model = Sequential()
    model.add(Dense(N_HIDDEN, batch_input_shape=(None, 6), activation='relu'))
    model.add(Dense(N_HIDDEN, activation='relu'))
    #model.add(Dense(N_HIDDEN, activation='relu'))
    model.add(MDN(OUTPUT_DIMS, N_MIXES))
    #model.compile(loss=get_mixture_loss_func(OUTPUT_DIMS,N_MIXES), optimizer=keras.optimizers.Adam())
    model.compile(loss=get_mixture_mse_accuracy(OUTPUT_DIMS,N_MIXES), optimizer=keras.optimizers.Adam())
    model.summary()
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=int(0.2*num_epochs))
    mc = ModelCheckpoint('./save/'+'best_'+model_file_name+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # fit model
    history = model.fit(x=x_input, y=y_input, batch_size=batch_size, epochs=num_epochs, validation_split=0.15, 
                        callbacks=[keras.callbacks.TerminateOnNaN(), es, mc])
    model.save('./save/'+model_file_name+'.h5')  # creates a HDF5 file 'my_model.h5'

    plt.figure(figsize=(10, 5))
    plt.title('mdn_{}_{}_{}_{}'.format(N_HIDDEN, N_MIXES, batch_size, num_epochs))
    maxLoss = np.max(history.history['loss'])*2.0
    plt.ylim([0,maxLoss])
    plt.plot(history.history['loss'], label = 'Train loss')
    plt.plot(history.history['val_loss'], label = 'Validation loss')
    plt.legend(('Train loss', 'Validation loss'))
    plt.xlabel('Episode')
    plt.ylabel('Loss (MSE)')
    plt.show()
    return model

def train2(x_input, img_input, y_input, args):
    N_HIDDEN = args.n_hidden
    N_MIXES = args.n_mixes
    OUTPUT_DIMS = args.output_dims
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    IMG_DIM = args.img_dim
    kRegularizer = args.kernel_regularizer
    dropOut = args.drop_out
    loadPrev = args.load_prev
    #
    strCurDateTime = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_file_name = 'mdn_32x5x2_32x5x2_32x3x2_{}_{}_{}_{}_{}_{}'.format(N_HIDDEN, N_MIXES, batch_size, num_epochs, args.loss_func, args.lr)
    #begin to build the model
    input1 = Input(shape=(6,), name='obs')
    input2 = Input(shape=(1,IMG_DIM,2*IMG_DIM), name='imgs')
    #first cnn to extract latent vector for imgs
    #conv = Conv2D(64, kernel_size=(3,3), strides=(2,2), data_format='channels_first', padding="valid", 
    #                activation="relu", kernel_regularizer=regularizers.l2(kRegularizer))(input2)
    #if args.use_batch_norm :
    #    conv = BatchNormalization(axis=1)(conv)
    #conv = Conv2D(32, kernel_size=(1,1), strides=(2,2), data_format='channels_first', padding="valid", 
    #                activation="relu", kernel_regularizer=regularizers.l2(kRegularizer))(conv)
    #if args.use_batch_norm :
    #    conv = BatchNormalization(axis=1)(conv)
    
    #20200505 add higher kernel size conv2d layers
    conv = Conv2D(32, kernel_size=(5,5), strides=(2,2), data_format='channels_first', padding="valid", 
                    activation="relu", kernel_regularizer=regularizers.l2(kRegularizer))(input2)
    if args.use_batch_norm :
        conv = BatchNormalization(axis=1)(conv)
    conv = Conv2D(32, kernel_size=(5,5), strides=(2,2), data_format='channels_first', padding="valid", 
                    activation="relu", kernel_regularizer=regularizers.l2(kRegularizer))(conv)
    if args.use_batch_norm :
        conv = BatchNormalization(axis=1)(conv)
    conv = Conv2D(32, kernel_size=(3,3), strides=(2,2), data_format='channels_first', padding="valid", 
                    activation="relu", kernel_regularizer=regularizers.l2(kRegularizer))(conv)
    if args.use_batch_norm :
        conv = BatchNormalization(axis=1)(conv)
    flatten1 = Flatten()(conv)
    #concatenate previous latent with input1
    merge = concatenate([input1, flatten1])
    if args.use_batch_norm :
        merge = BatchNormalization()(merge)
    dense = Dense(N_HIDDEN, activation='relu', kernel_regularizer=regularizers.l2(kRegularizer))(merge)
    if args.use_batch_norm :
        dense = BatchNormalization()(dense)
    dense = Dropout(dropOut)(dense)
    #dense = Dense(N_HIDDEN, activation='relu', kernel_regularizer=regularizers.l2(kRegularizer))(dense)
    #if args.use_batch_norm :
    #    dense = BatchNormalization()(dense)
    #dense = Dropout(dropOut)(dense)
    #dense = keras.layers.Dense(N_HIDDEN, activation='relu')(dense)
    mdn = MDN(OUTPUT_DIMS, N_MIXES)(dense)
    model = Model(inputs=[input1, input2], outputs=mdn)
    #model.compile(loss=get_mixture_loss_func(OUTPUT_DIMS,N_MIXES), optimizer=keras.optimizers.Adam())
    loss_func_name = 'mdn_loss_func'
    loss_func = get_mixture_loss_func(OUTPUT_DIMS,N_MIXES)
    if args.loss_func.lower() == 'mse':
        loss_func_name = 'mse_func'
        loss_func = get_mixture_mse_accuracy(OUTPUT_DIMS,N_MIXES)
    
    model.compile(loss=loss_func, optimizer=keras.optimizers.Adam(learning_rate=args.lr))
    model.summary()
    #try to load previous best model
    if loadPrev is not None and os.path.exists(loadPrev):
        model = load_model(loadPrev, custom_objects={'GlorotUniform': glorot_uniform(), 
                                                        'MDN': MDN(OUTPUT_DIMS,N_MIXES),
                                                        loss_func_name:loss_func})
        print('Succesfully load pre-trained model:', loadPrev)
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=int(0.2*num_epochs))
    mc = ModelCheckpoint('./save/best_'+model_file_name+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    logdir = "./logs/scalars/" + strCurDateTime
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    cb = [keras.callbacks.TerminateOnNaN(), es, mc]         #tensorboard_callback
    # fit model
    history = model.fit(x=[x_input, img_input], y=y_input, batch_size=batch_size, epochs=num_epochs, 
                        validation_split=0.2, callbacks=cb)
    best_val_loss = '{:.5f}'.format(np.min(history.history['val_loss']))
    model.save('./save/'+model_file_name+'_'+best_val_loss+'.h5')  # creates a HDF5 file 'my_model.h5'
    #now plot the training results
    plt.figure(figsize=(10, 5))
    plt.title(model_file_name)
    maxLoss = np.max(history.history['loss'])
    maxValLoss = np.max(history.history['val_loss'])
    maxLoss = max(maxLoss, maxValLoss)+0.2
    #plt.ylim([0,maxLoss])
    plt.plot(history.history['loss'], label = 'Train loss')
    plt.plot(history.history['val_loss'], label = 'Validation loss')
    plt.legend(('Train loss', 'Validation loss'))
    plt.xlabel('Episode')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.grid(axis='both', color='0.95', linestyle='-')
    #plt.show()
    plt.savefig('./save/'+model_file_name+'_'+best_val_loss+'_plot.png')
    #save history model fit data
    hist = np.concatenate(( np.expand_dims(history.history['loss'], axis=0), 
                            np.expand_dims(history.history['val_loss'], axis=0) ), axis=0)
    np.savetxt('./save/'+model_file_name+'_'+best_val_loss+'_hist.txt', hist, fmt = '%f')
    return model

def test(x_input, img_input, y_input, args, model=None):
    N_MIXES = args.n_mixes
    OUTPUT_DIMS = args.output_dims
    # load model from previous saved HDF5 file
    loss_func_name = 'mdn_loss_func'
    loss_func = get_mixture_loss_func(OUTPUT_DIMS,N_MIXES)
    if args.loss_func.lower() == 'mse':
        loss_func_name = 'mse_func'
        loss_func = get_mixture_mse_accuracy(OUTPUT_DIMS,N_MIXES)
    if model is None: 
        model_file_path = args.load_prev
        model_file_name = os.path.basename(model_file_path)
        model = load_model(model_file_path, custom_objects={'GlorotUniform': glorot_uniform(), 
                                                        'MDN': MDN(OUTPUT_DIMS,N_MIXES),
                                                        loss_func_name:loss_func})
    else:
        model_file_name = 'mdn2L_32x5x2_32x5x2_32x3x2_{}_{}_{}_{}_{}_{}'.format(args.n_hidden, args.n_mixes, args.batch_size, args.num_epochs, args.loss_func, args.lr)
    # summarize model.
    model.summary()
    ## Sample on some test data:
    x_test = x_input[0:600000,:]
    y_test = y_input[0:600000,:]
    print("Testing:", x_test.shape[0], "samples.")

    # Make predictions from the model
    y_predicted = model.predict([x_test, img_input])
    # y_predicted contains parameters for distributions, not actual points on the graph.
    # To find points on the graph, we need to sample from each distribution.

    # Split up the mixture parameters (for future fun)
    #mus = np.apply_along_axis((lambda a: a[:N_MIXES*OUTPUT_DIMS]), 1, y_predicted)
    #sigs = np.apply_along_axis((lambda a: a[N_MIXES*OUTPUT_DIMS:2*N_MIXES*OUTPUT_DIMS]), 1, y_predicted)
    #pis = np.apply_along_axis((lambda a: softmax(a[-N_MIXES:])), 1, y_predicted)

    # Sample from the predicted distributions
    y_samples = np.apply_along_axis(sample_from_output, 1, y_predicted, OUTPUT_DIMS, N_MIXES, temp=1.0, sigma_temp=1.0)
    y_samples = np.reshape(y_samples, [-1, OUTPUT_DIMS])
    print("Predict:", y_samples.shape, "samples.")
    #Calculate MSE loss
    rmseLoss = math.sqrt(((y_test - y_samples)**2).mean(axis=None))
    print('RMSE of prediction performance: {}'.format(rmseLoss))
    
    plt.figure(figsize=(10, 5))
    plt.title(model_file_name+'_MSE_validation')
    plt.plot(np.absolute(y_test - y_samples)[:,0],c='g',marker=(8,2,0),ls='--',label='Food Dist')
    plt.plot(np.absolute(y_test - y_samples)[:,1],c='r',marker="+",ls='-',label='Nest Dist')
    plt.legend(loc=2)
    plt.xlabel('samples')
    plt.ylabel('prediction error')
    #plt.show()
    plt.savefig('./save/{}_test_{:.4f}rmse_plot.png'.format(model_file_name, rmseLoss))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ma-foraging-mdn')
    parser.add_argument('--mode', type=str, default='train',help='Mode: train or test')
    parser.add_argument('--obs-data-file', type=str, default='./log/20200419_172857.csv',help='The obs input data file path')
    parser.add_argument('--imgs-data-file', type=str, default='./log/20200419_172857_msgtbl.json',help='The JSON imgs input data file path')
    parser.add_argument('--load-prev', type=str, default='',help='The pre-trained model to be loaded for continuing to train or test')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, constant or a schedule function')
    parser.add_argument('--n-hidden', type=int, default=256, help='Number of hidden neuron in Dense layers')
    parser.add_argument('--n-mixes', type=int, default=32, help='How many Gaussian distribution to be mixed to generate outputs')
    parser.add_argument('--output-dims', type=int, default=2, help='How many outputs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--num-epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--img-dim', type=int, default=24, help='Image size in one dimension default=24x24')
    parser.add_argument('--kernel-regularizer', type=float, default=0.001, help='kernel regularizer rate')
    parser.add_argument('--drop-out', type=float, default=0.25, help='drop out rate')
    parser.add_argument('--use-batch-norm', type=int, default=1, help='default to use batch normalization (set 0 to off)')
    parser.add_argument('--loss-func', type=str, default='mse',help='The loss function: mse or log_loss')
    ###################################################################
    args = parser.parse_args()
    #x_input, y_input = loadData1('./log/rehearsalFeatures_20200325_023237.csv')
    nRecords = 600000
    x_input, y_input = loadData1(args.obs_data_file)
    img_input = loadData2(args.imgs_data_file, nRecords)
    if args.mode == 'test':
        test(x_input, img_input, y_input, args)
    else:
        #train1(x_input, y_input)
        #loadPrev=None
        X = x_input[0:nRecords,:]
        X_img = img_input
        Y = y_input[0:nRecords,:]
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        X_img = X_img[indices]
        Y = Y[indices]
        print(X.shape, X_img.shape, Y.shape)
        model = train2(X, X_img, Y, args)
        test(x_input, img_input, y_input, args, model)
    