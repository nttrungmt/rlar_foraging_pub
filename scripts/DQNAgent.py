import random
import gym
import numpy as np
from collections import deque
from per_memory import Memory
#from keras import backend
#from keras.models import Sequential, Model
#from keras.layers import Dense, Input, Conv2D, Flatten, concatenate
#from keras.optimizers import Adam, RMSprop
#import keras.backend as K
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K

#20200514
from mdn import *
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, buffer_size=102400,
                    epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99,
                    hidden_num=256, use_per=True, lr_decay_rate=0.0, use_batch_norm=False,
                    use_neighbor_image=False, imgDim=24, cnn_layers_str=None, #20200410 - 20200415
                    hidden_features_mode=0, mdn_model_params=None):    #20200514
        self.state_size = state_size
        self.state_img_size = 0
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.use_per = use_per
        self.lr_decay_rate = lr_decay_rate
        self.nUpdates = 0
        #20200514 add code to mask the hidden features based on the mode
        self.hidden_features_mode = hidden_features_mode
        self.mdn_model_params = mdn_model_params
        self.mdn_model = None
        #load pretrained mdn model to generate hidden features from visible features
        if self.hidden_features_mode == 2 or self.hidden_features_mode == 3:
            N_MIXES = self.mdn_model_params['n_mixes']
            OUTPUT_DIMS = self.mdn_model_params['output_dims']
            loss_type = self.mdn_model_params['loss_func']
            model_file_path = self.mdn_model_params['model_file_path']
            loss_func_name = 'mdn_loss_func'
            loss_func = get_mixture_loss_func(OUTPUT_DIMS,N_MIXES)
            if loss_type.lower() == 'mse':
                loss_func_name = 'mse_func'
                loss_func = get_mixture_mse_accuracy(OUTPUT_DIMS,N_MIXES)
            self.mdn_model = load_model(model_file_path, 
                                        custom_objects={'GlorotUniform': glorot_uniform(), 
                                                        'MDN': MDN(OUTPUT_DIMS,N_MIXES),
                                                        loss_func_name:loss_func})
        #20200514 -------------------------------------------------------
        self.use_neighbor_image = use_neighbor_image
        self.IMG_DIM = imgDim
        if self.use_neighbor_image :
            self.state_img_size = 3*2*self.IMG_DIM*self.IMG_DIM
            self.state_size = state_size - self.state_img_size
        if not use_per:
            self.memory = deque(maxlen=buffer_size)     #reaplay buffer for offline training
        else:
            self.memory = Memory(buffer_size)
        self.learning_rate = learning_rate          #learning rate
        self.cur_lr = self.learning_rate
        self.gamma = gamma                          #discount rate
        self.epsilon_start = epsilon_start          #starting exploration prob
        self.epsilon = epsilon_start                #current exploration prob
        self.epsilon_min = epsilon_min              #ending exploration prob
        self.epsilon_decay = epsilon_decay          #exploration decay rate
        self.hidden_num = hidden_num                #number of hidden neurons at each hidden layer
        self.cnn_layers_str = cnn_layers_str        #cnn architecture: [ [kernel, filter, stride], [kernel, filter, stride], ... ]
        self.cnn_layers = None
        self.use_batch_norm = use_batch_norm        #option to use batch normalization
        self.model = self._build_model() if not self.use_neighbor_image else self._build_model2()            #train model
        self.target_model = self._build_model() if not self.use_neighbor_image else self._build_model2()     #target model
        self.updateTargetNet()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.hidden_num, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.hidden_num, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_model2(self):
        # Neural Net for Deep-Q learning Model
        input1 = Input(shape=(self.state_size,), name='obs')                               #3*13 if use full hidden features
        input2 = Input(shape=(3,self.IMG_DIM,2*self.IMG_DIM), name='imgs')
        # CNN for processing neigbor stacked images
        #print('cnn_layers=', self.cnn_layers)
        if self.cnn_layers_str is None or self.cnn_layers_str.strip()=='':
            if self.IMG_DIM == 64:
                conv = Conv2D(32, kernel_size=(8,8), strides=(4,4), data_format='channels_first', padding="valid", activation="relu")(input2)
                conv = Conv2D(64, kernel_size=(4,4), strides=(2,2), data_format='channels_first', padding="valid", activation="relu")(conv)
                conv = Conv2D(64, kernel_size=(3,3), strides=(1,1), data_format='channels_first', padding="valid", activation="relu")(conv)
                self.cnn_layers = [[32,8,4], [64,4,2], [64,3,1]]
            elif self.IMG_DIM == 32:
                conv = Conv2D(32, kernel_size=(4,4), strides=(2,2), data_format='channels_first', padding="valid", activation="relu")(input2)
                conv = Conv2D(64, kernel_size=(3,3), strides=(2,2), data_format='channels_first', padding="valid", activation="relu")(conv)
                self.cnn_layers = [[32,4,2], [64,3,2]]
            else:
                #24x24 20200413
                #conv = Conv2D(32, kernel_size=(8,8), strides=(4,4), data_format='channels_first', padding="valid", activation="relu")(input2)
                #conv = Conv2D(64, kernel_size=(4,4), strides=(2,2), data_format='channels_first', padding="valid", activation="relu")(conv)
                #self.cnn_layers = [[32,8,4], [64,4,2]]
                #24x24 20200415
                if False:
                    conv = Conv2D(64, kernel_size=(3,3), strides=(2,2), data_format='channels_first', padding="valid", activation="relu")(input2)
                    conv = Conv2D(32, kernel_size=(1,1), strides=(2,2), data_format='channels_first', padding="valid", activation="relu")(conv)
                    self.cnn_layers = [[64,3,2], [32,1,2]]
                else:
                    conv = Conv2D(32, kernel_size=(5,5), strides=(2,2), data_format='channels_first', padding="valid", activation="relu")(input2)
                    conv = Conv2D(32, kernel_size=(5,5), strides=(2,2), data_format='channels_first', padding="valid", activation="relu")(conv)
                    conv = Conv2D(32, kernel_size=(3,3), strides=(2,2), data_format='channels_first', padding="valid", activation="relu")(conv)
                    self.cnn_layers = [[32,5,2], [32,5,2], [32,3,2]]
        else:
            arr = eval(self.cnn_layers_str.split()[0])
            assert(len(arr)>0)
            self.cnn_layers = arr
            for i in range(len(arr)):
                conv = Conv2D(arr[i][0], kernel_size=arr[i][1], strides=arr[i][2], 
                                data_format='channels_first', padding="valid", activation="relu")(input2 if i==0 else conv)
        flatten1 = Flatten()(conv)
        # merge CNN result with old obs input
        merge = concatenate([input1, flatten1])
        #dense = Dense(self.hidden_num, activation='relu')(merge)
        #dense = Dense(self.hidden_num, activation='relu')(dense)
        if(isinstance(self.hidden_num, list)):
            for idx in range(len(self.hidden_num)):
                dense = Dense(self.hidden_num[idx], activation='relu')(merge if idx==0 else dense)
        else:
            dense = Dense(self.hidden_num, activation='relu')(merge)
        dense = Dense(self.action_size, activation='linear')(dense)
        #finally create the model from custom inputs and other layers
        model = Model(inputs=[input1, input2], outputs=dense)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #model.compile(loss="mean_squared_error",
        #                   optimizer=RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01))
        return model

    def maskHiddenFeatures(self,obs,imgs=None):
        masked_obs = obs.copy()
        if self.hidden_features_mode == 1:      #not use hidden features => masked to 0
            masked_obs[:,10] = 0    #mask foodDist column to 0
            masked_obs[:,11] = 0    #mask nestDist column to 0
        elif self.hidden_features_mode == 2:    #generate hidden features by pretrained MDN model
            N_MIXES = self.mdn_model_params['n_mixes']
            OUTPUT_DIMS = self.mdn_model_params['output_dims']
            x_test = obs[:,3:9]
            imgs_test = imgs[:,0,:,:]
            if len(imgs_test.shape) < 4:
                imgs_test = np.expand_dims(imgs_test, axis=0)
            y_predicted = self.mdn_model.predict([x_test, imgs_test])
            # Split up the mixture parameters (for future fun)
            #mus = np.apply_along_axis((lambda a: a[:N_MIXES*OUTPUT_DIMS]), 1, y_predicted)
            #sigs = np.apply_along_axis((lambda a: a[N_MIXES*OUTPUT_DIMS:2*N_MIXES*OUTPUT_DIMS]), 1, y_predicted)
            #pis = np.apply_along_axis((lambda a: softmax(a[-N_MIXES:])), 1, y_predicted)
            y_samples = np.apply_along_axis(sample_from_output, 1, y_predicted, OUTPUT_DIMS, N_MIXES, temp=1.0, sigma_temp=1.0)
            y_samples = np.reshape(y_samples, [-1, OUTPUT_DIMS])
            # copy to masked array
            masked_obs[:,10] = y_samples[:,0]    #mask foodDist column to with the first column of y_samples
            masked_obs[:,11] = y_samples[:,1]    #mask nestDist column to with the second column of y_samples
        return masked_obs
    
    def mdnIncLearning(self,state):
        if self.hidden_features_mode == 3:    #generate hidden features by pretrained MDN model
            N_MIXES = self.mdn_model_params['n_mixes']
            OUTPUT_DIMS = self.mdn_model_params['output_dims']
            sArr = np.asarray([s['obs'] for s in state])
            sImgArr = np.asarray([s['imgs'] for s in state])
            x_test = sArr[:,3:9]
            y_test = sArr[:,10:12]
            imgs_test = sImgArr[:,0,:,:]
            if len(imgs_test.shape) < 4:
                imgs_test = np.expand_dims(imgs_test, axis=1)
            history = self.mdn_model.fit(x=[x_test, imgs_test], y=y_test, epochs=1, validation_split = 0.2)
            return np.mean(history.history['loss'])
    
    def maskHiddenFeaturesDict(self,state):
        err = 0
        if self.hidden_features_mode == 1:      #not use hidden features => masked to 0
            for s in state:
                s['obs'][10] = 0    #mask foodDist column to 0
                s['obs'][11] = 0    #mask nestDist column to 0
        elif self.hidden_features_mode == 2 or self.hidden_features_mode == 3:    #generate hidden features by pretrained MDN model
            N_MIXES = self.mdn_model_params['n_mixes']
            OUTPUT_DIMS = self.mdn_model_params['output_dims']
            sArr = np.asarray([s['obs'] for s in state])
            sImgArr = np.asarray([s['imgs'] for s in state])
            x_test = sArr[:,3:9]
            imgs_test = sImgArr[:,0,:,:]
            if len(imgs_test.shape) < 4:
                imgs_test = np.expand_dims(imgs_test, axis=1)
            y_predicted = self.mdn_model.predict([x_test, imgs_test])
            # Split up the mixture parameters (for future fun)
            #mus = np.apply_along_axis((lambda a: a[:N_MIXES*OUTPUT_DIMS]), 1, y_predicted)
            #sigs = np.apply_along_axis((lambda a: a[N_MIXES*OUTPUT_DIMS:2*N_MIXES*OUTPUT_DIMS]), 1, y_predicted)
            #pis = np.apply_along_axis((lambda a: softmax(a[-N_MIXES:])), 1, y_predicted)
            y_samples = np.apply_along_axis(sample_from_output, 1, y_predicted, OUTPUT_DIMS, N_MIXES, temp=1.0, sigma_temp=1.0)
            y_samples = np.reshape(y_samples, [-1, OUTPUT_DIMS])
            # copy to masked array
            errArr = np.zeros((len(state),2))
            for idx in range(len(state)):
                errArr[idx][0] = abs(state[idx]['obs'][10]-y_samples[idx][0])
                errArr[idx][1] = abs(state[idx]['obs'][11]-y_samples[idx][1])
                state[idx]['obs'][10] = y_samples[idx][0]    #mask foodDist column to with the first column of y_samples
                state[idx]['obs'][11] = y_samples[idx][1]    #mask nestDist column to with the second column of y_samples
            err = np.mean(errArr)
        return err

    def memorize(self, state, action, reward, next_state, done):
        if not self.use_per:
            self.memory.append((state, action, reward, next_state, done))
        else:
            s = np.reshape(state['obs'], [-1,self.state_size])
            s_ = np.reshape(next_state['obs'], [-1,self.state_size])
            if not self.use_neighbor_image:
                target = self.model.predict_on_batch(s)
                old_val = target[0][action]
                target_val = self.target_model.predict_on_batch(s_)
            else:
                imgs_s = np.reshape(state['imgs'], [-1,3,self.IMG_DIM,2*self.IMG_DIM])
                imgs_s_ = np.reshape(next_state['imgs'], [-1,3,self.IMG_DIM,2*self.IMG_DIM])
                #20200514 masked hidden features if need
                #s = self.maskHiddenFeatures(s, imgs_s)
                #s_ = self.maskHiddenFeatures(s_, imgs_s_)
                #20200514 ------------------------------
                target = self.model.predict_on_batch([s, imgs_s])
                old_val = target[0][action]
                target_val = self.target_model.predict_on_batch([s_, imgs_s_])
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.argmax(target_val)
            error = abs(old_val - target[0][action])
            self.memory.add(error, (state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            actType = 0     #EXPLORE
            #acts = [random.randrange(self.action_size) for _ in range(state.shape[0])]
            acts = [random.randrange(self.action_size) for _ in range(len(state))]
        else:
            actType = 1     #EXPLOIT
            #acts = np.argmax(self.model.predict_on_batch(state), axis=1).tolist()
            if not self.use_neighbor_image:
                sArr = np.asarray([s['obs'] for s in state])
                assert sArr.shape == (len(state), self.state_size)
                acts = np.argmax(self.model.predict_on_batch(sArr), axis=1).tolist()
            else:
                sArr = np.asarray([s['obs'] for s in state])
                sImgArr = np.asarray([s['imgs'] for s in state])
                assert sArr.shape == (len(state), self.state_size) and sImgArr.shape == (len(state),3,self.IMG_DIM,2*self.IMG_DIM)
                #20200514 masked hidden features if need
                #sArr = self.maskHiddenFeatures(sArr, sImgArr)
                #20200514 ------------------------------
                acts = np.argmax(self.model.predict_on_batch([sArr, sImgArr]), axis=1).tolist()
        return acts, actType        # returns action

    def replay(self, batch_size, nUpdate=1):
        self.nUpdates += 1
        histLoss = []
        for _ in range(nUpdate):
            idxs = None
            if not self.use_per:
                minibatch = random.sample(self.memory, batch_size)
            else:
                minibatch, idxs, _ = self.memory.sample(batch_size)
            states = np.array([i[0]['obs'] for i in minibatch])
            actions = np.array([i[1] for i in minibatch])
            rewards = np.array([i[2] for i in minibatch])
            next_states = np.array([i[3]['obs'] for i in minibatch])
            dones = np.array([i[4] for i in minibatch])
            #now update target val 
            if not self.use_neighbor_image:
                target = self.model.predict(states)
                target_next = self.model.predict(next_states)         #DQN
                target_val = self.target_model.predict(next_states)   #Target model
            else:
                imgs_states = np.array([i[0]['imgs'] for i in minibatch])
                imgs_next_states = np.array([i[3]['imgs'] for i in minibatch])
                #20200514 masked hidden features if need
                #states = self.maskHiddenFeatures(states, imgs_states)
                #next_states = self.maskHiddenFeatures(next_states, imgs_next_states)
                #20200514 ------------------------------
                target = self.model.predict([states, imgs_states])
                target_next = self.model.predict([next_states, imgs_next_states])         #DQN
                target_val = self.target_model.predict([next_states, imgs_next_states])   #Target model
            errors = np.zeros(batch_size)
            for i in range(batch_size):
                oldVal = target[i][actions[i]]
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    # selection of action is from model update is from target model
                    a = np.argmax(target_next[i])
                    target[i][actions[i]] = rewards[i] + self.gamma * (target_val[i][a])
                if self.use_per:
                    errors[i] = abs(oldVal - target[i][actions[i]])
                    self.memory.update(idxs[i], errors[i])
            if not self.use_neighbor_image:
                hist = self.model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)
            else:
                hist = self.model.fit([states, imgs_states], target, batch_size=batch_size, epochs=1, verbose=0)
            histLoss.extend(hist.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            #self.epsilon -= self.epsilon_decay
        #2020328 decay learning rate
        if self.lr_decay_rate > 0:
            self.cur_lr = self.learning_rate - self.lr_decay_rate * (self.nUpdates-1)
            K.set_value(self.model.optimizer.lr, self.cur_lr)
        return np.mean(histLoss)

    def updateTargetNet(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
