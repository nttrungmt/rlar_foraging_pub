import os
import sys
import math
import random
import argparse
from datetime import datetime, date
from collections import deque
import multiprocessing
import numpy as np
import tensorflow as tf
#import cProfile
import logger
import tensorboard_logging
#from geometry_msgs.msg import Pose2D
from ArgosForagingEnvironment import ArgosForagingEnvironment, Pose2D
from HoffAgent import HoffAgent
from DQNAgent import DQNAgent

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from shutil import copyfile

#SELECT_AGENT = 1    #0: Hoff; 1: DQN
EPISODE_MEAN_REWARD = "eps_mean_reward"
NUM_FOOD_RETURN = "num_food_return"
WINAVG_NUM_FOOD_RETURN = "num_food_return_winavg"
CUMAVG_NUM_FOOD_RETURN = "num_food_return_cumavg"
QVAL_LOSS = "qval_loss"
WINDOW_SIZE = 20
SAVE_BEST_FREQ = 10     #the frequency to save best model so far, every 10 epochs
MAX_MODELS_KEEP = 10    #keep the last 10 best models

def trainWithHoff(params, args):
    #load parameters
    env = params['env']
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    NUM_EPISODES = args.num_episodes
    STEPS_PER_EPISODE = args.steps_per_episode
    #create agent
    agent = HoffAgent(state_size, action_size)
    #create logger
    f_name = "Hoff"
    log_path = params['log_path']
    start_time = datetime.now()
    dt_string = start_time.strftime("%Y%m%d_%H%M%S")
    sess_name = '{}_{}_{}'.format(dt_string, NUM_EPISODES, STEPS_PER_EPISODE)
    tb_base_log_path = os.path.join(log_path, "tf_board", f_name, sess_name)
    if(os.path.exists(args.restore_log_path)):
        tb_base_log_path = args.restore_log_path
    logger.configure(tb_base_log_path, ['stdout', 'log', 'csv'])      #['stdout', 'log', 'csv']
    #else:
    #    print('Could not find the restore_log_path. Check the parameter again!')
    #    return
    tb_logger_eprewmean = tensorboard_logging.Logger(os.path.join(tb_base_log_path, EPISODE_MEAN_REWARD))
    tb_logger_nFoodReturn = tensorboard_logging.Logger(os.path.join(tb_base_log_path, NUM_FOOD_RETURN))
    tb_logger_nFoodReturn_winavg = tensorboard_logging.Logger(os.path.join(tb_base_log_path, WINAVG_NUM_FOOD_RETURN))
    tb_logger_nFoodReturn_cumavg = tensorboard_logging.Logger(os.path.join(tb_base_log_path, CUMAVG_NUM_FOOD_RETURN))
    #number of returned food file
    exp_nFoodReturn_path = str(os.path.join(tb_base_log_path, 'Handcoded_0.csv'))
    exp_nFoodReturn_file = open(exp_nFoodReturn_path, 'a+')
    exp_nFoodReturn_file.write('#Hoff_{}\n'.format(sess_name))
    #create the list to store EPISODE MEAN REWARD
    eps_mean_reward = []
    eps_food_return = []
    winavg_nFoodReturn = []
    cumavg_nFoodReturn = []
    for ep_count  in range(NUM_EPISODES):
        episode_rew = 0
        _, state = env.reset()
        #if(ep_count==0):
        #    print("ep={} state.shape={}".format(ep_count, state.shape))
        #state = np.reshape(state, [-1, state_size])
        ep_start_time = datetime.now()
        for time in range(STEPS_PER_EPISODE):
            # env.render()
            action = agent.act(state)        #20200409 adapt because state now is a dictionary with two parts: old obs, and new 'imgs' observation
            _, next_state, reward, done, _ = env.step(action)
            #if(ep_count == 0):
            #    print('ep {} time {} reward=\n{}'.format(ep_count, time, \
            #            [['{:02d}'.format(idx), '{:.3f}'.format(reward[idx])] for idx in range(len(reward))]))
            episode_rew += np.sum(reward)
            #next_state = np.reshape(next_state, [-1, state_size])
            #if(time==0):
            #    print("ep={} time={} action.len={} s'.shape={} elapse_time={}".format(\
            #            ep_count, time, len(action), next_state.shape, (datetime.now() - ep_start_time)))
            state = next_state
            if time >= STEPS_PER_EPISODE-1:
                totalFoodReturn = 0
                for idx in range(0, env.num_robots):
                    totalFoodReturn += next_state[idx]['obs'][9]
                eps_mean_reward.append(episode_rew/STEPS_PER_EPISODE)
                eps_food_return.append(totalFoodReturn)
                print("ep: {}/{}, mean_avg_reward: {:.3f}, total_food_return: {}, exec_time= {}".format( \
                    ep_count , NUM_EPISODES, eps_mean_reward[-1], eps_food_return[-1], (datetime.now() - ep_start_time)))
                #write tensorboard logging
                cur_ep_cnt = ep_count + args.last_timesteps
                tb_logger_eprewmean.log_scalar(EPISODE_MEAN_REWARD, eps_mean_reward[-1], cur_ep_cnt)
                tb_logger_eprewmean.flush()
                tb_logger_nFoodReturn.log_scalar(NUM_FOOD_RETURN, eps_food_return[-1], cur_ep_cnt)
                tb_logger_nFoodReturn.flush()
                # +++++++ 20200412 add more log for winavg and cumavg
                winavg_nFoodReturn.append(np.mean(eps_food_return[-WINDOW_SIZE:]))
                if ep_count == 0:
                    cumavg_nFoodReturn.append(eps_food_return[-1])
                else:
                    cumavg_nFoodReturn.append((eps_food_return[-1]+ep_count*cumavg_nFoodReturn[-1])/(ep_count+1))
                tb_logger_nFoodReturn_winavg.log_scalar(WINAVG_NUM_FOOD_RETURN, winavg_nFoodReturn[-1], cur_ep_cnt)
                tb_logger_nFoodReturn_winavg.flush()
                tb_logger_nFoodReturn_cumavg.log_scalar(CUMAVG_NUM_FOOD_RETURN, cumavg_nFoodReturn[-1], cur_ep_cnt)
                tb_logger_nFoodReturn_cumavg.flush()
                #
                exp_nFoodReturn_file.write('{},{}\n'.format(cur_ep_cnt, eps_food_return[-1]))
                exp_nFoodReturn_file.flush()
                if ep_count % args.log_interval == 0:
                    #ev = explained_variance(values, returns)
                    logger.logkv("epoch_index", cur_ep_cnt)
                    #logger.logkv("serial_timesteps", cur_ep_cnt*STEPS_PER_EPISODE)
                    logger.logkv("total_timesteps", cur_ep_cnt*STEPS_PER_EPISODE*20)
                    logger.logkv("fps", (ep_count*STEPS_PER_EPISODE/(datetime.now() - start_time).total_seconds()))
                    #logger.logkv("explained_variance", float(ev))
                    logger.logkv('eps_mean_reward', eps_mean_reward[-1])
                    logger.logkv('eps_food_return', eps_food_return[-1])
                    logger.logkv('eps_food_return_winavg', winavg_nFoodReturn[-1])
                    logger.logkv('eps_food_return_cumavg', cumavg_nFoodReturn[-1])
                    logger.logkv('time_elapsed', (datetime.now() - start_time).total_seconds()/3600)
                    logger.dumpkvs()
                # ------- 20200412
    print("Finish executing Hoff Agent with {} episodes in {}".format(NUM_EPISODES, (datetime.now() - start_time)))
    exp_nFoodReturn_file.close()
    #20200613 draw the plot of winavg and cumavg
    plt.figure(figsize=(10, 5))
    plt.title(f_name+'_'+sess_name)
    #plt.ylim([0,maxLoss])
    plt.plot(winavg_nFoodReturn, label = 'winavg')
    plt.plot(cumavg_nFoodReturn, label = 'cumavg')
    plt.legend(('winavg', 'cumavg'))
    plt.xlabel('Episode')
    plt.ylabel('Number of food return')
    #plt.yscale('log')
    plt.grid(axis='both', color='0.95', linestyle='-')
    #plt.show()
    plot_file_path = str(os.path.join(tb_base_log_path, f_name+'_'+sess_name+'_plot.png'))
    plt.savefig(plot_file_path)

def trainWithDQN(params, args):
    #load parameters
    env = params['env']
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    NUM_EPISODES = args.num_episodes
    STEPS_PER_EPISODE = args.steps_per_episode
    batch_size = args.batch_size
    #create agent and logger
    lr_decay_rate = 0.0
    total_train_steps = NUM_EPISODES*STEPS_PER_EPISODE/args.replay_interval
    if(args.lr_decay):
        lr_decay_rate = args.lr/total_train_steps
    #20200421 will effectively overwrite the epsilon_decay, will be based on the exploration_rate parameter settings
    total_exploration_steps = total_train_steps*args.exploration_rate
    #epsilon_decay = (args.epsilon_start - args.epsilon_min)/total_exploration_steps
    epsilon_decay = math.pow(args.epsilon_min/args.epsilon_start, 1/total_exploration_steps)
    hidden_size = eval(args.hidden_size.split()[0])
    mdn_model_params = {'n_mixes':args.n_mixes, 'output_dims':2, 'loss_func':args.mdn_loss_func, 'model_file_path':args.mdn_model_path}
    agent = DQNAgent(state_size, action_size, learning_rate=args.lr, gamma=args.gamma, buffer_size=args.buffer_size, \
                    epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min, epsilon_decay=epsilon_decay, \
                    hidden_num=hidden_size, use_per = args.use_per, lr_decay_rate=lr_decay_rate,
                    use_neighbor_image = args.use_neighbor_image, imgDim=env.IMG_DIM, cnn_layers_str=args.cnn_layers,
                    hidden_features_mode=args.hidden_features_mode, mdn_model_params=mdn_model_params)
    #load previous saved model if specified
    if os.path.isfile(args.restore_model_path):
        agent.load(args.restore_model_path)
        print('Load previous saved model from this path: {}'.format(args.restore_model_path))
    #else:
    #    print('Could not find the restore_model_path. Check the parameter again!')
    #    return
    #create tensorboard logger
    f_name = "DQN"
    log_path = params['log_path']
    mdl_log_path = str(os.path.join(params['log_path'], f_name))
    if not os.path.exists(mdl_log_path):
        os.makedirs(mdl_log_path)
    #save_path = params['save_path']
    start_time = datetime.now()
    dt_string = start_time.strftime("%Y%m%d_%H%M%S")       #
    hfMode = 'FullHF'
    if args.hidden_features_mode == 1:
        hfMode = 'NoHF'
    elif args.hidden_features_mode == 2 or args.hidden_features_mode == 3:
        hfMode = 'mdnHF'
    prefix_name = '{}_{}_{}_{}'.format(hfMode, args.lr, str(agent.cnn_layers).replace('[','').replace(']','').replace(', ',''), args.hidden_size)
    if len(args.model_prefix_name)>0:
        prefix_name = args.model_prefix_name
    model_file_name = '{}_{}'.format(prefix_name, dt_string)
    tb_base_log_path = os.path.join(log_path, "tf_board", f_name, model_file_name)
    if os.path.exists(args.restore_log_path):
        tb_base_log_path = args.restore_log_path
        print('Continue to write log to this folder: {}'.format(args.restore_log_path))
    logger.configure(tb_base_log_path, ['stdout', 'log', 'csv'])      #['stdout', 'log', 'csv']
    #else:
    #    print('Could not find the restore_log_path. Check the parameter again!')
    #    return
    tb_logger_eprewmean = tensorboard_logging.Logger(os.path.join(tb_base_log_path, EPISODE_MEAN_REWARD))
    tb_logger_nFoodReturn = tensorboard_logging.Logger(os.path.join(tb_base_log_path, NUM_FOOD_RETURN))
    tb_logger_nFoodReturn_winavg = tensorboard_logging.Logger(os.path.join(tb_base_log_path, WINAVG_NUM_FOOD_RETURN))
    tb_logger_nFoodReturn_cumavg = tensorboard_logging.Logger(os.path.join(tb_base_log_path, CUMAVG_NUM_FOOD_RETURN))
    tb_logger_nFoodReturn_qvalloss = tensorboard_logging.Logger(os.path.join(tb_base_log_path, QVAL_LOSS))
    #save the final model parameters
    #model_file_name = '{}_{}_{}_{}_{}_{}'.format(prefix_name, NUM_EPISODES + args.last_timesteps, STEPS_PER_EPISODE, batch_size, args.hidden_size, dt_string)
    model_params_path = str(os.path.join(mdl_log_path, model_file_name+'_params.txt'))
    with open(model_params_path, 'w') as f:
        f.write('==================================================\n')
        f.write("agent={}\n".format('DQN'))
        f.write("num_robots={}\n".format(args.num_robots))
        f.write("port={}\n".format(args.port))
        f.write("num_episodes={}\n".format(NUM_EPISODES))
        f.write("norm_data={}\n".format(args.norm_data))
        f.write("use_neighbor_image={}\n".format(args.use_neighbor_image))
        #f.write("use_neighbor_rewards={}\n".format(args.use_neighbor_rewards))
        f.write("local_team_reward_rate={}\n".format(args.local_team_reward_rate))
        f.write("global_reward_rate={}\n".format(args.global_reward_rate))
        f.write("img_dim={}\n".format(env.IMG_DIM))
        f.write("steps_per_episode={}\n".format(STEPS_PER_EPISODE))
        f.write("hidden_features_mode={}\n".format(args.hidden_features_mode))
        f.write("n_mixes={}\n".format(args.n_mixes))
        f.write("mdn_loss_func={}\n".format(args.mdn_loss_func))
        f.write("mdn_model_path={}\n".format(args.mdn_model_path))
        f.write("batch_size={}\n".format(batch_size))
        f.write("num_batch_update_steps={}\n".format(args.num_batch_update_steps))
        f.write("use_per={}\n".format(args.use_per))
        f.write("hidden_size={}\n".format(args.hidden_size))
        f.write("cnn_layers={}\n".format(agent.cnn_layers))
        f.write("lr={}\n".format(args.lr))
        f.write("lr_decay={}\n".format(args.lr_decay))
        f.write("gamma={}\n".format(args.gamma))
        f.write("epsilon_start={}\n".format(args.epsilon_start))
        f.write("epsilon_min={}\n".format(args.epsilon_min))
        f.write("exploration_rate={}\n".format(args.exploration_rate))
        f.write("epsilon_decay={}\n".format(epsilon_decay))
        f.write("buffer_size={}\n".format(args.buffer_size))
        f.write("learning_starts={}\n".format(args.learning_starts))
        f.write("replay_interval={}\n".format(args.replay_interval))
        f.write("model_prefix_name={}\n".format(args.model_prefix_name))
        f.write("restore_model_path={}\n".format(args.restore_model_path))
        f.write("restore_log_path={}\n".format(args.restore_log_path))
        f.write('model.summary\n')
        agent.model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write('==================================================\n')
    #number of returned food file
    exp_nFoodReturn_path = str(os.path.join(tb_base_log_path, prefix_name+'_'+dt_string+'_nFoodRt.csv'))
    exp_nFoodReturn_file = open(exp_nFoodReturn_path, 'a+')
    exp_nFoodReturn_file.write('#{}\n'.format(model_file_name))
    # list to store the EPISODE MEAN REWARD
    eps_mean_reward = []
    eps_food_return = []
    winavg_nFoodReturn = []
    cumavg_nFoodReturn = []
    bestMeanReward = -100
    bestMeanFoodReturn = -100
    bestSavedModeList = []
    for ep_count in range(NUM_EPISODES):
        episode_rew = 0
        episode_rew_new = 0
        _, state = env.reset()
        #if(ep_count==0):
        #    print("ep={} state.shape={}".format(ep_count, state.shape))
        #state = np.reshape(state, [-1, state_size])
        if ep_count == 0:
            with open(model_params_path, 'a') as f:
                if env.m_foraging_params is not None:
                    f.write('env.m_foraging_params\n')
                    for key,value in env.m_foraging_params.items():
                        f.write("{}={}\n".format(key,value))
            tfb_model_params_path = str(os.path.join(tb_base_log_path, model_file_name+'_params.txt'))
            copyfile(model_params_path, tfb_model_params_path)
        ep_start_time = datetime.now()
        action = None
        actTypes = []
        histLoss = []
        mdnErr = []
        mdnIncLearningErr=[]
        for time in range(STEPS_PER_EPISODE):
            # env.render()
            #20200518 maskHiddenStates if need
            mdnMeanErr = agent.maskHiddenFeaturesDict(state)
            if args.hidden_features_mode == 2 or args.hidden_features_mode == 3:
                mdnErr.append(mdnMeanErr)
            #20200518 ------------------------
            action, actType = agent.act(state)
            actTypes.append(actType)
            _, next_state, reward, done, infos = env.step(action)
            #++++++++ 20200419 change the way to calculate global reward
            episode_rew += np.sum(infos['orgRewards'])
            episode_rew_new += np.sum(reward)
            #--------
            #next_state = np.reshape(next_state, [-1, state_size])
            #add to DQN buffer
            for idx in range(0, env.num_robots):
                agent.memorize(state[idx], action[idx], reward[idx], next_state[idx], done[idx])
            state = next_state
            #20200706 online learning mdn
            if (time+1)%args.replay_interval == 0:
                mdnIncLearningErr.append(agent.mdnIncLearning(state))
            if time >= STEPS_PER_EPISODE-1:
                totalFoodReturn = 0
                for idx in range(0, env.num_robots):
                    totalFoodReturn += next_state[idx]['obs'][9]
                eps_food_return.append(totalFoodReturn)
                eps_mean_reward.append(episode_rew/STEPS_PER_EPISODE)
                #print("ep: {}/{}, actType: {}".format(ep_count , NUM_EPISODES, actTypes))
                print("ep: {}/{}, meanAvgReward: {:.3f}, newMeanRwd={:.3f}, nFoodReturn: {}, exec_time= {}".format( \
                    ep_count, NUM_EPISODES, eps_mean_reward[-1], episode_rew_new/STEPS_PER_EPISODE, eps_food_return[-1], \
                    (datetime.now() - ep_start_time)))
                #write tensorboard logging
                cur_ep_cnt = ep_count + args.last_timesteps
                tb_logger_eprewmean.log_scalar(EPISODE_MEAN_REWARD, eps_mean_reward[-1], cur_ep_cnt)
                tb_logger_eprewmean.flush()
                tb_logger_nFoodReturn.log_scalar(NUM_FOOD_RETURN, eps_food_return[-1], cur_ep_cnt)
                tb_logger_nFoodReturn.flush()
                # +++++++ 20200412 add more log for winavg and cumavg
                winavg_nFoodReturn.append(np.mean(eps_food_return[-WINDOW_SIZE:]))
                if ep_count == 0:
                    cumavg_nFoodReturn.append(eps_food_return[-1])
                else:
                    cumavg_nFoodReturn.append((eps_food_return[-1]+ep_count*cumavg_nFoodReturn[-1])/(ep_count+1))
                mean_qval_loss = np.mean(histLoss) if len(histLoss) > 0 else 0
                #mean_qval_loss = np.mean(histLoss[-100:]) if len(histLoss) > 100 else 0
                tb_logger_nFoodReturn_winavg.log_scalar(WINAVG_NUM_FOOD_RETURN, winavg_nFoodReturn[-1], cur_ep_cnt)
                tb_logger_nFoodReturn_winavg.flush()
                tb_logger_nFoodReturn_cumavg.log_scalar(CUMAVG_NUM_FOOD_RETURN, cumavg_nFoodReturn[-1], cur_ep_cnt)
                tb_logger_nFoodReturn_cumavg.flush()
                tb_logger_nFoodReturn_qvalloss.log_scalar(QVAL_LOSS, mean_qval_loss, cur_ep_cnt)
                tb_logger_nFoodReturn_qvalloss.flush()
                #
                exp_nFoodReturn_file.write('{},{}\n'.format(cur_ep_cnt, eps_food_return[-1]))
                exp_nFoodReturn_file.flush()
                if ep_count % args.log_interval == 0:
                    #ev = explained_variance(values, returns)
                    logger.logkv("epoch_index", cur_ep_cnt)
                    #logger.logkv("serial_timesteps", cur_ep_cnt*STEPS_PER_EPISODE)
                    logger.logkv("total_timesteps", cur_ep_cnt*STEPS_PER_EPISODE)
                    logger.logkv("fps", (ep_count*STEPS_PER_EPISODE/(datetime.now() - start_time).total_seconds()))
                    #logger.logkv("explained_variance", float(ev))
                    logger.logkv('eps_mean_reward', eps_mean_reward[-1])
                    logger.logkv('eps_food_return', eps_food_return[-1])
                    logger.logkv('eps_food_return_winavg', winavg_nFoodReturn[-1])
                    logger.logkv('eps_food_return_cumavg', cumavg_nFoodReturn[-1])
                    logger.logkv('eps_mean_qval_loss', mean_qval_loss)
                    logger.logkv('time_elapsed', (datetime.now() - start_time).total_seconds()/3600)
                    logger.logkv('cur_lr', agent.cur_lr)
                    logger.logkv('cur_epsilon', agent.epsilon)
                    if args.hidden_features_mode == 2:
                        logger.logkv('eps_mean_mdn_error', np.mean(mdnErr))
                    if args.hidden_features_mode == 3:
                        logger.logkv('eps_mean_mdn_error', np.mean(mdnErr))
                        logger.logkv('eps_mean_mdn_inc_learn_loss', np.mean(mdnIncLearningErr))
                    logger.dumpkvs()
                # ------- 20200412
            #update DQN model if there are enough samples and replay frequency
            if (ep_count*STEPS_PER_EPISODE+time+1)*env.num_robots > args.learning_starts and (time+1)%args.replay_interval == 0:
                qval_loss = agent.replay(batch_size, args.num_batch_update_steps)
                histLoss.append(qval_loss)
        #update target model of dqn after each episode
        agent.updateTargetNet()
        if (ep_count+1) % args.save_interval == 0:
            model_file_path = str(os.path.join(mdl_log_path, model_file_name+'.h5'))
            agent.save(model_file_path)
        if (ep_count+1) % SAVE_BEST_FREQ == 0:
            curMeanReward = np.mean(eps_mean_reward[-WINDOW_SIZE:])
            curMeanFoodReturn = np.mean(eps_food_return[-WINDOW_SIZE:])
            print('ep {}-{} mean avg reward={:.3f} mean food return={}'.format(
                    ep_count-SAVE_BEST_FREQ+1, ep_count, curMeanReward, curMeanFoodReturn))
            #try to keep the best model
            if(curMeanReward + curMeanFoodReturn) > (bestMeanReward + bestMeanFoodReturn):
                bestMeanReward = curMeanReward
                bestMeanFoodReturn = curMeanFoodReturn
                bestMdlSuffix = '_{}eps_{:.3f}_{}.h5'.format(ep_count + args.last_timesteps, curMeanReward, curMeanFoodReturn)
                model_file_path = str(os.path.join(mdl_log_path, model_file_name+bestMdlSuffix))
                if len(bestSavedModeList) >= MAX_MODELS_KEEP:
                    path_to_delete = bestSavedModeList[0]
                    if os.path.exists(path_to_delete):
                        os.remove(path_to_delete)
                    del bestSavedModeList[0]
                bestSavedModeList.append(model_file_path)
                agent.save(model_file_path)
    print("Finish train DQN Agent mode={}, num-robots={} with {} episodes in {}".format(hfMode, args.num_robots, NUM_EPISODES, (datetime.now() - start_time)))
    exp_nFoodReturn_file.close()
    #20200613 draw the plot of winavg and cumavg
    plt.figure(figsize=(10, 5))
    plt.title(model_file_name)
    #plt.ylim([0,maxLoss])
    plt.plot(winavg_nFoodReturn, label = 'winavg')
    plt.plot(cumavg_nFoodReturn, label = 'cumavg')
    plt.legend(('winavg', 'cumavg'))
    plt.xlabel('Episode')
    plt.ylabel('Number of food return')
    #plt.yscale('log')
    plt.grid(axis='both', color='0.95', linestyle='-')
    #plt.show()
    plot_file_path = str(os.path.join(tb_base_log_path, model_file_name+'_plot.png'))
    plt.savefig(plot_file_path)

def testWithDQN(params, args):
    #load parameters
    env = params['env']
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    NUM_EPISODES = args.num_episodes
    STEPS_PER_EPISODE = args.steps_per_episode
    batch_size = args.batch_size
    #create agent and logger
    lr_decay_rate = 0.0
    total_train_steps = NUM_EPISODES*STEPS_PER_EPISODE/args.replay_interval
    if(args.lr_decay):
        lr_decay_rate = args.lr/total_train_steps
    #20200421 will effectively overwrite the epsilon_decay, will be based on the exploration_rate parameter settings
    total_exploration_steps = total_train_steps*args.exploration_rate
    #epsilon_decay = (args.epsilon_start - args.epsilon_min)/total_exploration_steps
    epsilon_decay = math.pow(args.epsilon_min/args.epsilon_start, 1/total_exploration_steps)
    hidden_size = eval(args.hidden_size.split()[0])
    mdn_model_params = {'n_mixes':args.n_mixes, 'output_dims':2, 'loss_func':args.mdn_loss_func, 'model_file_path':args.mdn_model_path}
    agent = DQNAgent(state_size, action_size, learning_rate=args.lr, gamma=args.gamma, buffer_size=args.buffer_size, \
                    epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=epsilon_decay, \
                    hidden_num=hidden_size, use_per = args.use_per, lr_decay_rate=lr_decay_rate,
                    use_neighbor_image = args.use_neighbor_image, imgDim=env.IMG_DIM, cnn_layers_str=args.cnn_layers,
                    hidden_features_mode=args.hidden_features_mode, mdn_model_params=mdn_model_params)
    #load previous saved model if specified
    if os.path.isfile(args.restore_model_path):
        agent.load(args.restore_model_path)
        print('Load previous saved model from this path: {}'.format(args.restore_model_path))
    else:
        print('Could not find the restore_model_path. Check the parameter again!')
        return
    #create tensorboard logger
    f_name = "DQN"
    log_path = params['log_path']
    mdl_log_path = str(os.path.join(params['log_path'], f_name))
    if not os.path.exists(mdl_log_path):
        os.makedirs(mdl_log_path)
    #save_path = params['save_path']
    start_time = datetime.now()
    dt_string = start_time.strftime("%Y%m%d_%H%M%S")       #
    hfMode = 'FullHF'
    if args.hidden_features_mode == 1:
        hfMode = 'NoHF'
    elif args.hidden_features_mode == 2:
        hfMode = 'mdnHF'
    prefix_name = 'test_{}_{}_{}'.format(args.num_robots, NUM_EPISODES, dt_string)
    if len(args.model_prefix_name)>0:
        prefix_name = args.model_prefix_name
    #model_file_name = '{}_{}'.format(prefix_name, dt_string)
    model_file_full_name = os.path.basename(args.restore_model_path)
    (model_file_name, ext) = os.path.splitext(model_file_full_name)
    tb_base_log_path = os.path.join(log_path, "tf_board", f_name, model_file_name, prefix_name)
    if os.path.exists(args.restore_log_path):
        tb_base_log_path = args.restore_log_path
        print('Continue to write log to this folder: {}'.format(args.restore_log_path))
    logger.configure(tb_base_log_path, ['stdout', 'log', 'csv'])      #['stdout', 'log', 'csv']
    #else:
    #    print('Could not find the restore_log_path. Check the parameter again!')
    #    return
    tb_logger_eprewmean = tensorboard_logging.Logger(os.path.join(tb_base_log_path, EPISODE_MEAN_REWARD))
    tb_logger_nFoodReturn = tensorboard_logging.Logger(os.path.join(tb_base_log_path, NUM_FOOD_RETURN))
    tb_logger_nFoodReturn_winavg = tensorboard_logging.Logger(os.path.join(tb_base_log_path, WINAVG_NUM_FOOD_RETURN))
    tb_logger_nFoodReturn_cumavg = tensorboard_logging.Logger(os.path.join(tb_base_log_path, CUMAVG_NUM_FOOD_RETURN))
    #tb_logger_nFoodReturn_qvalloss = tensorboard_logging.Logger(os.path.join(tb_base_log_path, QVAL_LOSS))
    #save the final model parameters
    #model_file_name = '{}_{}_{}_{}_{}_{}'.format(prefix_name, NUM_EPISODES + args.last_timesteps, STEPS_PER_EPISODE, batch_size, args.hidden_size, dt_string)
    model_params_path = str(os.path.join(tb_base_log_path, prefix_name+'_params.txt'))
    with open(model_params_path, 'w') as f:
        f.write('==================================================\n')
        f.write("agent={}\n".format('DQN'))
        f.write("num_robots={}\n".format(args.num_robots))
        f.write("num_episodes={}\n".format(NUM_EPISODES))
        f.write("norm_data={}\n".format(args.norm_data))
        f.write("use_neighbor_image={}\n".format(args.use_neighbor_image))
        f.write("img_dim={}\n".format(env.IMG_DIM))
        f.write("steps_per_episode={}\n".format(STEPS_PER_EPISODE))
        f.write("hidden_features_mode={}\n".format(args.hidden_features_mode))
        f.write("n_mixes={}\n".format(args.n_mixes))
        f.write("mdn_loss_func={}\n".format(args.mdn_loss_func))
        f.write("mdn_model_path={}\n".format(args.mdn_model_path))
        f.write("restore_model_path={}\n".format(args.restore_model_path))
        f.write('==================================================\n')
    #number of returned food file
    exp_nFoodReturn_path = str(os.path.join(tb_base_log_path, prefix_name+'_0.csv'))
    exp_nFoodReturn_file = open(exp_nFoodReturn_path, 'a+')
    exp_nFoodReturn_file.write('#{},{}\n'.format(model_file_name, prefix_name))
    # list to store the EPISODE MEAN REWARD
    eps_mean_reward = []
    eps_food_return = []
    winavg_nFoodReturn = 0
    cumavg_nFoodReturn = 0
    bestMeanReward = -100
    bestMeanFoodReturn = -100
    bestSavedModeList = []
    for ep_count in range(NUM_EPISODES):
        episode_rew = 0
        episode_rew_new = 0
        _, state = env.reset()
        #if(ep_count==0):
        #    print("ep={} state.shape={}".format(ep_count, state.shape))
        #state = np.reshape(state, [-1, state_size])
        if ep_count == 0:
            with open(model_params_path, 'a') as f:
                if env.m_foraging_params is not None:
                    f.write('env.m_foraging_params\n')
                    for key,value in env.m_foraging_params.items():
                        f.write("{}={}\n".format(key,value))
        ep_start_time = datetime.now()
        action = None
        actTypes = []
        histLoss = []
        mdnErr = []
        for time in range(STEPS_PER_EPISODE):
            # env.render()
            #20200518 maskHiddenStates if need
            mdnMeanErr = agent.maskHiddenFeaturesDict(state)
            if args.hidden_features_mode == 2:
                mdnErr.append(mdnMeanErr)
            #20200518 ------------------------
            action, actType = agent.act(state)
            actTypes.append(actType)
            _, next_state, reward, done, infos = env.step(action)
            #++++++++ 20200419 change the way to calculate global reward
            episode_rew += np.sum(infos['orgRewards'])
            episode_rew_new += np.sum(reward)
            #--------
            #next_state = np.reshape(next_state, [-1, state_size])
            #add to DQN buffer
            #for idx in range(0, env.num_robots):
            #    agent.memorize(state[idx], action[idx], reward[idx], next_state[idx], done[idx])
            state = next_state
            if time >= STEPS_PER_EPISODE-1:
                totalFoodReturn = 0
                for idx in range(0, env.num_robots):
                    totalFoodReturn += next_state[idx]['obs'][9]
                eps_food_return.append(totalFoodReturn)
                eps_mean_reward.append(episode_rew/STEPS_PER_EPISODE)
                #print("ep: {}/{}, actType: {}".format(ep_count , NUM_EPISODES, actTypes))
                print("ep: {}/{}, meanAvgReward: {:.3f}, newMeanRwd={:.3f}, nFoodReturn: {}, exec_time= {}".format( \
                    ep_count, NUM_EPISODES, eps_mean_reward[-1], episode_rew_new/STEPS_PER_EPISODE, eps_food_return[-1], \
                    (datetime.now() - ep_start_time)))
                #write tensorboard logging
                cur_ep_cnt = ep_count + args.last_timesteps
                tb_logger_eprewmean.log_scalar(EPISODE_MEAN_REWARD, eps_mean_reward[-1], cur_ep_cnt)
                tb_logger_eprewmean.flush()
                tb_logger_nFoodReturn.log_scalar(NUM_FOOD_RETURN, eps_food_return[-1], cur_ep_cnt)
                tb_logger_nFoodReturn.flush()
                # +++++++ 20200412 add more log for winavg and cumavg
                winavg_nFoodReturn = np.mean(eps_food_return[-WINDOW_SIZE:])
                cumavg_nFoodReturn = (eps_food_return[-1]+ep_count*cumavg_nFoodReturn)/(ep_count+1)
                #mean_qval_loss = np.mean(histLoss) if len(histLoss) > 0 else 0
                #mean_qval_loss = np.mean(histLoss[-100:]) if len(histLoss) > 100 else 0
                tb_logger_nFoodReturn_winavg.log_scalar(WINAVG_NUM_FOOD_RETURN, winavg_nFoodReturn, cur_ep_cnt)
                tb_logger_nFoodReturn_winavg.flush()
                tb_logger_nFoodReturn_cumavg.log_scalar(CUMAVG_NUM_FOOD_RETURN, cumavg_nFoodReturn, cur_ep_cnt)
                tb_logger_nFoodReturn_cumavg.flush()
                #tb_logger_nFoodReturn_qvalloss.log_scalar(QVAL_LOSS, mean_qval_loss, cur_ep_cnt)
                #tb_logger_nFoodReturn_qvalloss.flush()
                #
                exp_nFoodReturn_file.write('{},{}\n'.format(cur_ep_cnt, eps_food_return[-1]))
                exp_nFoodReturn_file.flush()
                if ep_count % args.log_interval == 0:
                    #ev = explained_variance(values, returns)
                    logger.logkv("epoch_index", cur_ep_cnt)
                    #logger.logkv("serial_timesteps", cur_ep_cnt*STEPS_PER_EPISODE)
                    logger.logkv("total_timesteps", cur_ep_cnt*STEPS_PER_EPISODE)
                    logger.logkv("fps", (ep_count*STEPS_PER_EPISODE/(datetime.now() - start_time).total_seconds()))
                    #logger.logkv("explained_variance", float(ev))
                    logger.logkv('eps_mean_reward', eps_mean_reward[-1])
                    logger.logkv('eps_food_return', eps_food_return[-1])
                    logger.logkv('eps_food_return_winavg', winavg_nFoodReturn)
                    logger.logkv('eps_food_return_cumavg', cumavg_nFoodReturn)
                    #logger.logkv('eps_mean_qval_loss', mean_qval_loss)
                    logger.logkv('time_elapsed', (datetime.now() - start_time).total_seconds()/3600)
                    #logger.logkv('cur_lr', agent.cur_lr)
                    #logger.logkv('cur_epsilon', agent.epsilon)
                    if args.hidden_features_mode == 2:
                        logger.logkv('eps_mean_mdn_error', np.mean(mdnErr))
                    logger.dumpkvs()
                # ------- 20200412
    print("Finish test DQN Agent mode={}, num-robots={} with {} episodes in {}".format(hfMode, args.num_robots, NUM_EPISODES, (datetime.now() - start_time)))
    exp_nFoodReturn_file.close()

def main():
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})
    ######### declare argument parser ######################################################
    parser = argparse.ArgumentParser(description='ma-foraging')
    parser.add_argument('--agent', type=str, default='Hoff',help='Agent to train: Hoff | DQN')
    parser.add_argument('--mode', type=str, default='train',help='Train or test')
    parser.add_argument('--num-robots', type=int, default=40, help='Number of robots')
    parser.add_argument('--port', type=int, default=12345, help='Communication port between Argos Simulator and Python')
    parser.add_argument('--dumpExps', action='store_true', help='Dump the experiences and observations to JSON/CSV')
    parser.add_argument('--norm-data', action='store_true', help='Normalize observation data or not')
    parser.add_argument('--use-neighbor-image', action='store_true', help='Generate neighbor cardinality images for observation')       #20200409 add neighbor image observation
    #parser.add_argument('--use-neighbor-rewards', action='store_true', help='Take into account of neighbor rewards or not')             #20200419 add neighbor rewards
    parser.add_argument('--local-team-reward-rate', type=float, default=0.0, help='The rate of local neighbor rewards to be used')    #20200424 change from flag to rate (0.0 => no use)
    parser.add_argument('--global-reward-rate', type=float, default=0.0, help='The rate of global  rewards to be used')    #20200424 add rate of global reward (0.0 => no use)
    parser.add_argument('--img-dim', type=int, default=24, metavar='pixel', help='The width/height of neighbor image')
    parser.add_argument('--num-episodes', type=int, default=3000, metavar='EPISODES', help='Number of training episodes')
    parser.add_argument('--steps-per-episode', type=int, default=150, metavar='STEPS', help='number of steps per each episode')
    parser.add_argument('--hidden-features-mode', type=int, default=0, help='Mode to use hidden features: 0-full, 1-not use, 2-MDN, 3-MDN incremental learning')
    # Parameters for DQN
    parser.add_argument('--batch-size', type=int, default=256, help='number of samples per update')
    parser.add_argument('--use-per', action='store_true', help='Use prioritize experience replay in DQN or not')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, constant or a schedule function')
    parser.add_argument('--lr-decay', action='store_true', help='Decreasing the learning rate over the time or not')
    parser.add_argument('--gamma', type=float, default=0.99, help='discounting factor')
    parser.add_argument('--exploration-rate', type=float, default=0.8, help='exploration rate, default=10% of training process will drop from epsilon_start to epsilon_min')
    #parser.add_argument('--epsilon-decay', type=float, default=0.99, help='epsilon decay rate')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Starting epsilon')
    parser.add_argument('--epsilon-min', type=float, default=0.015, help='Starting epsilon')
    parser.add_argument('--hidden-size', type=str, default='256', help='Size of one hidden layer (DQN), i.e. 64 or [256,128]')
    parser.add_argument('--cnn-layers', type=str, default='', help='The configurations of cnn layers, i.e. [[64,3,2], [32,1,2]]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='The optimizer to train the model (Adam(default) | RMSprop | Adamax')
    parser.add_argument('--use-batch-norm', action='store_true', help='Use batch normalization or not')
    parser.add_argument('--buffer-size', type=int, default=100240, help='DQN replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=50000, help='how many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--replay-interval', type=int, default=10, metavar='STEPS', help='Number of steps between DQN replay (training) calls')
    # Parameters to log, save/load models
    parser.add_argument('--base-path', type=str, default='.', help='Base path to write log, final model, tensor board')
    parser.add_argument('--log-interval', type=int, default=1, metavar='STEPS', help='Number of episodes between logging events')
    parser.add_argument('--save-interval', type=int, default=5, metavar='SIZE', help='Number of episodes between saving events')
    parser.add_argument('--model-prefix-name', type=str, default='', help='The prefix label to put in model file name')
    #parser.add_argument('--restore', action='store_true', help='Restore previously saved model or train new')
    parser.add_argument('--restore-model-path', type=str, default='', help='Previous saved model path to load')
    parser.add_argument('--restore-log-path', type=str, default='', help='Previous log path to use')
    parser.add_argument('--last-timesteps', type=int, default=0, metavar='STEPS', help='Last training time steps')
    #parser.add_argument('--evaluate', action='store_true', help='Evaluate only, no training, must provided saved model')
    #parser.add_argument('--evaluate-episodes', type=int, default=10, metavar='EPISODES', help='Number of evaluation episodes')
    parser.add_argument('--n-mixes', type=int, default=2, help='number of mixes in the MDN model')
    parser.add_argument('--mdn-loss-func', type=str, default='mse', help='Loss func, default=mse')
    parser.add_argument('--mdn-model-path', type=str, default='', help='Previous trained MDN model path')
    # Parameters for LSTM
    parser.add_argument('--beta', type=float, default=0.001, help='actor critic beta parameter')
    parser.add_argument('--lr-a', type=float, default=0.000001, help='actor learning rate')
    parser.add_argument('--lr-c', type=float, default=0.00001, help='critic learning rate')
    parser.add_argument('--lstm-input-size', type=int, default=200, help='LSTM input size')
    parser.add_argument('--num-batch-update-steps', type=int, default=4, help='Number of DQN/LSTM batch update times')
    parser.add_argument('--restore-actor-model-path', type=str, default='', help='Previous saved actor model (LSTM) path to load')
    parser.add_argument('--restore-critic-model-path', type=str, default='', help='Previous saved critic model (LSTM) path to load')
    ###################################################################
    args = parser.parse_args()
    ################### perform training ##############################
    train(args)

if __name__ == '__main__':
    main()
