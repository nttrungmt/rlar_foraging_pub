import math
import time
import numpy as np
#import rospy
#from ma_foraging.srv import *
#from geometry_msgs.msg import Pose2D
import gym
from gym import spaces
import socket
import struct
import json
import logging
import os
from datetime import datetime, date
from utils import dump_json
import csv
from PIL import Image, ImageDraw
import sys, random

USE_ROS_THREAD = False
MAX_COMM_RADIUS = 200       #the max range of communication radius: 200cm
MAX_DISTANCE = 45           #the max distance between two robots: 45m
MAX_CARDINALITY = 15        #the max (infinity) value of food/nest cardinality
MAX_GRAY_VAL = 200          #max gray value (0-255) to transfrom food/nest cardinality to gray value

class Pose2D():
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta

class AIServiceResponse:
    def __init__(self):
        self.observations = None
        self.rewards = None
        self.dones = None
        #self.ids = None

class ArgosForagingEnvironment(gym.Env):
    """A tensorflow openai based environment for the Argos robotics simulator."""
    def __init__(self, start_poses, service_name = 'AIService', port=12345, dumpExps=False, dumpFolder=None, 
                    normalizeData=False, use_neighbor_image=False, imgDim=24, 
                    localTeamRwdRate=0.0, globalRwdRate=0.0):      #useNeighborRwd=False
        """The length of the start poses must match and determine the number of robots.
        :param start_poses: The desired start poses of the robots."""
        if USE_ROS_THREAD:
            self.service = rospy.ServiceProxy(service_name, AIService)
        else:
            # Create a TCP/IP socket
            self.port = port
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Bind the socket to the port
            self.server_address = ('0.0.0.0', port)       #port=12345 by default
            print('[Python] starting up on %s port %s' % self.server_address)
            self.sock.bind(self.server_address)
            # Listen for incoming connections
            self.sock.listen(1)
            self.connection = None
            while self.connection == None:
                # Wait for a connection
                print('[Python] waiting for a connection')
                self.connection, self.client_address = self.sock.accept()
            print('[Python] connection from', self.client_address)
        self.__size_message_length = 8
        self.num_robots = len(start_poses)
        self.start_poses = start_poses
        self.current_response = None
        self.num_envs = 1
        self.m_episode_time = 0
        self.m_foraging_params = None   #20200420
        self.m_loop_cnt = 0
        self.dumpExps = dumpExps
        self.dumpFolder = dumpFolder
        self.normalizeData = normalizeData
        self.use_neighbor_image = use_neighbor_image
        self.IMG_DIM = imgDim
        #self.useNeighborRwd = useNeighborRwd
        self.localTeamRwdRate = localTeamRwdRate
        self.globalRwdRate = globalRwdRate
        ####################################################
        self.action_space = spaces.Discrete(2)
        if not self.use_neighbor_image:
            self.observation_space = spaces.Box(low=0, high=255, shape=(13*3,))      ##20200228 10 -> 12; 20200229 12->15
        else:
            ##20200228 10 -> 12; 20200229 12->15 ; 20200409: add 2 image 64x64
            self.observation_space = spaces.Box(low=0, high=255, shape=(13*3+3*2*self.IMG_DIM*self.IMG_DIM,))
        if(self.dumpFolder is not None and not os.path.exists(self.dumpFolder)):
            os.makedirs(self.dumpFolder)
        cur_time = datetime.now()
        dt_string = cur_time.strftime("%Y%m%d_%H%M%S")
        self.json_file_path = os.path.join(self.dumpFolder, dt_string+'.json')
        self.csv_file_path = os.path.join(self.dumpFolder, dt_string+'.csv')
        if(self.dumpExps):
            self.csvfile = open(self.csv_file_path, 'w+', newline='')
            self.fieldnames = ['minDist', 'maxDist', 'foodClosest', 'nestClosest', 'foodFarthest', 'nestFarthest','foodDist', 'nestDist']
            self.csvwriter = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames, lineterminator=os.linesep)
            self.csvwriter.writeheader()
        else:
            self.csvwriter = None
        #+++++ 20200419 add json to store list of msgTbl
        self.msgtbl_json_file = os.path.join(self.dumpFolder, dt_string+'_msgtbl.json')
        #----- 20200419
        self.prevState = None
        self.prev_obs_t1 = [None]*self.num_robots
        self.prev_obs_t2 = [None]*self.num_robots
        self.prev_img_t1 = [None]*self.num_robots
        self.prev_img_t2 = [None]*self.num_robots

    def __del__(self):
        if self.csvwriter is not None:
            self.csvfile.close()
        if not USE_ROS_THREAD:
            self.sock.close()
    
    def generateNeighborImage1(self, ob):
        imgF = Image.new('L', (self.IMG_DIM,self.IMG_DIM), color=255)
        imgN = Image.new('L', (self.IMG_DIM,self.IMG_DIM), color=255)
        idrawF = ImageDraw.Draw(imgF)
        idrawN = ImageDraw.Draw(imgN)
        lstPts = ob['msgTbl']
        if lstPts is None:
            lstPts = []
        rScale = 0.16/4*self.IMG_DIM
        for i in range(len(lstPts)):
            x0 = self.IMG_DIM/2*(1 + lstPts[i]['dist']/MAX_COMM_RADIUS*math.sin(math.radians(90)-lstPts[i]['angle']))
            y0 = self.IMG_DIM/2*(1 - lstPts[i]['dist']/MAX_COMM_RADIUS*math.sin(lstPts[i]['angle']))
            c0 = int(lstPts[i]['fHop']*MAX_GRAY_VAL/MAX_CARDINALITY)
            c1 = int(lstPts[i]['nHop']*MAX_GRAY_VAL/MAX_CARDINALITY)
            #print(x0,y0,c0,c1)
            left = round(x0-rScale/2+1)
            xright = round(xleft+rScale)-1
            ytop = round(y0-rScale/2)
            ybottom = round(ytop+rScale)-1
            idrawF.ellipse([xleft, ytop, xright, ybottom], fill=c0, width=1)
            idrawN.ellipse([xleft, ytop, xright, ybottom], fill=c1, width=1)
            #print(x0,y0,c0,c1,' => ',xleft,xright,ytop,ybottom)
        #return imgF.tobytes(), imgN.tobytes()
        arr1 = np.array(list(imgF.tobytes()))
        arr2 = np.array(list(imgN.tobytes()))
        #print(arr1.shape, type(arr1), arr2.shape, type(arr2))
        arr = np.hstack((arr1, arr2))
        arr = arr.astype(np.uint8)
        #print(arr.shape, type(arr))
        return arr

    def generateNeighborImage2(self, ob):
        img = Image.new('L', (2*self.IMG_DIM,self.IMG_DIM), color=255)
        idraw = ImageDraw.Draw(img)
        lstPts = ob['msgTbl']
        if lstPts is None:
            lstPts = []
        rScale = 0.16/4*self.IMG_DIM
        for i in range(len(lstPts)):
            x0 = (self.IMG_DIM/2-1)*(1 + lstPts[i]['dist']/MAX_COMM_RADIUS*math.sin(math.radians(90)-lstPts[i]['angle']))
            y0 = self.IMG_DIM/2*(1 - lstPts[i]['dist']/MAX_COMM_RADIUS*math.sin(lstPts[i]['angle']))
            c0 = int(lstPts[i]['fHop']*MAX_GRAY_VAL/MAX_CARDINALITY)
            c1 = int(lstPts[i]['nHop']*MAX_GRAY_VAL/MAX_CARDINALITY)
            #print(x0,y0,c0,c1)
            xleft = round(x0-rScale/2+1)
            xright = round(xleft+rScale)-1
            ytop = round(y0-rScale/2)
            ybottom = round(ytop+rScale)-1
            idraw.rectangle([xleft,              ytop, xright,              ybottom], fill=c0, width=1)
            idraw.rectangle([xleft+self.IMG_DIM, ytop, xright+self.IMG_DIM, ybottom], fill=c1, width=1)
            #print(x0,y0,c0,c1,' => (',xleft,ytop,'),(',xright,ybottom,')')
        arr = np.array(list(img.tobytes()))
        #print(arr.shape, type(arr))
        #arr = arr.astype(np.uint8)
        arr = np.reshape(arr.astype(np.uint8), (self.IMG_DIM,2*self.IMG_DIM,1))
        arr = np.moveaxis(arr, 2, 0)
        #print(arr.shape, type(arr))
        return arr

    def single_ob_to_list(self, ob):
        values = []
        values.append(ob['curState'])
        #values.append(ob['walkerPersistenceCnt'])
        #values.append(ob['beaconPersistenceCnt'])
        values.append(ob['numBeacons']/(self.num_robots if self.normalizeData else 1))
        values.append(ob['hasFood'])
        values.append(ob['minDist']/(MAX_COMM_RADIUS if self.normalizeData else 1))
        values.append(ob['maxDist']/(MAX_COMM_RADIUS if self.normalizeData else 1))
        values.append(ob['foodClosest']/(MAX_CARDINALITY if self.normalizeData else 1))
        values.append(ob['nestClosest']/(MAX_CARDINALITY if self.normalizeData else 1))
        values.append(ob['foodFarthest']/(MAX_CARDINALITY if self.normalizeData else 1))
        values.append(ob['nestFarthest']/(MAX_CARDINALITY if self.normalizeData else 1))
        values.append(ob['numFoodReturn'])
        values.append(ob['foodDist']/(MAX_DISTANCE if self.normalizeData else 1))               #20200228
        values.append(ob['nestDist']/(MAX_DISTANCE if self.normalizeData else 1))               #20200228
        values.append(ob['numWalkers']/(self.num_robots if self.normalizeData else 1))          #20200229
        #values.append(ob['fHop'])                   #20200229
        #values.append(ob['nHop'])                   #20200229
        return values

    def observation_to_list(self, observations, init=False):
        arrObs = []
        if init:
            for idx in range(0, self.num_robots):
                ob = observations[idx]
                values = np.asarray(self.single_ob_to_list(ob)).flatten()
                #if self.m_loop_cnt == 0:
                #    print('init idx={} shape={}'.format(idx, values.shape))
                self.prev_obs_t1[idx] = values
                self.prev_obs_t2[idx] = values
                if self.use_neighbor_image :
                    img = self.generateNeighborImage2(ob)
                    self.prev_img_t1[idx] = img
                    self.prev_img_t2[idx] = img
        for idx in range(0, self.num_robots):
            ob = observations[idx]
            values = np.asarray(self.single_ob_to_list(ob)).flatten()
            obs = np.hstack((values, self.prev_obs_t1[idx], self.prev_obs_t2[idx]))
            #if self.m_loop_cnt == 0:
            #    print('idx={} ob.shape={}, obs.shape={}'.format(idx, values.shape, obs.shape))
            self.prev_obs_t2[idx] = self.prev_obs_t1[idx]
            self.prev_obs_t1[idx] = values
            assert obs.shape==(13*3,)
            #++++++ 20200409
            #arrObs.append(obs)
            if self.use_neighbor_image:
                img = self.generateNeighborImage2(ob)
                #imgs = np.vstack((img, self.prev_img_t1[idx], self.prev_img_t2[idx]))
                #imgs = np.reshape(imgs, (3, 2*self.IMG_DIM, self.IMG_DIM))
                imgs = np.concatenate((img, self.prev_img_t1[idx], self.prev_img_t2[idx]), axis=0)
                self.prev_img_t2[idx] = self.prev_img_t1[idx]
                self.prev_img_t1[idx] = img
                arrObs.append({'obs': obs, 'imgs': imgs})
                if(idx < 5 and self.m_loop_cnt % 3000 == 0):
                    #arrs_ = np.moveaxis(imgs, 0, 2)
                    arrs_ = np.reshape(imgs, (3*self.IMG_DIM, 2*self.IMG_DIM))
                    img_3 = Image.fromarray(arrs_)
                    img_3.save('./log/test_argos_3imgs_{}_{}.bmp'.format(self.port, idx))
                assert imgs.shape==(3, self.IMG_DIM, 2*self.IMG_DIM)
            else:
                arrObs.append({'obs': obs})
            #------ 20200409
        return arrObs

    def step(self, actions):
        if np.isnan(actions[0]):
            exit()
        service_success = False
        service_exception = False
        while not service_success:
            try:
                #print "[Python] Service call, wait for response at " + str(self.m_episode_time)
                if USE_ROS_THREAD:
                    response = self.service(actions, [])
                    service_success = True
                    if service_exception:
                        # If we got an exception we abort the trajectory, as the simulation might be in an inconstant state.
                        response.done = [True] * self.num_robots
                else:
                    response = self.sockService(actions, [])
                    service_success = True
            except Exception as e:
                print("[Python] Service call failed. Trying again in 3 seconds: " + str(e))
                service_exception = True
                time.sleep(3)
        self.current_response = response
        self.m_episode_time += 1
        self.m_loop_cnt += 1
        #dump json if activate
        if(self.dumpExps):
            #exp_dt = {}
            #exp_dt['states'] = self.prevState
            #exp_dt['actions'] = actions
            #exp_dt['rewards'] = response.rewards
            #exp_dt['next_states'] = response.observations
            #exp_dt['dones'] = response.dones
            #dump_json(exp_dt, self.json_file_path, overwrite=False)
            #dump csv for rehearsal features
            for idx in range(0, self.num_robots):
                ob = response.observations[idx]
                self.csvwriter.writerow({ 'minDist':'{:.4f}'.format(ob['minDist']), \
                                          'maxDist':'{:.4f}'.format(ob['maxDist']), \
                                          'foodClosest':ob['foodClosest'], \
                                          'nestClosest':ob['nestClosest'], \
                                          'foodFarthest':ob['foodFarthest'], \
                                          'nestFarthest':ob['nestFarthest'], \
                                          'foodDist':'{:.4f}'.format(ob['foodDist']), \
                                          'nestDist':'{:.4f}'.format(ob['nestDist']) })
                dump_json({'msgTbl': ob['msgTbl']}, self.msgtbl_json_file, overwrite=False)
        #now update prevState for next round
        self.prevState = response.observations
        #+++++++++++ 20200418 add global reward scale
        rewards = np.asarray(response.rewards)
        new_rewards = rewards.copy()
        assert(rewards.shape == (self.num_robots,))
        #if(self.useNeighborRwd):
        if self.globalRwdRate > 0:      #20200424
            new_rewards += self.globalRwdRate*np.sum(rewards)/self.num_robots
        if self.localTeamRwdRate > 0:   #20200424
            for idx in range(0, self.num_robots):
                ob = response.observations[idx]
                lstPts = ob['msgTbl']
                if lstPts is None:
                    lstPts = []
                neighbor_rewards = 0
                nb_count = 0
                for i in range(len(lstPts)):
                    nb_id = lstPts[i]['id']
                    neighbor_rewards += (MAX_COMM_RADIUS-lstPts[i]['dist'])/MAX_COMM_RADIUS*rewards[nb_id]
                    nb_count += 1
                if nb_count > 0:
                    new_rewards[idx] += self.localTeamRwdRate*neighbor_rewards/nb_count
        #----------- 20200418
        return range(0, self.num_robots), self.observation_to_list(response.observations), \
               new_rewards, \
               np.asarray(response.dones), \
               {'orgRewards': rewards}

    def reset(self):
        service_success = False
        while not service_success:
            try:
                if USE_ROS_THREAD:
                    response = self.service(list(), self.start_poses)
                else:
                    response = self.sockService(list(), self.start_poses)
                service_success = True
            except Exception as e:
                print("[Python] Service call failed. Trying again in 3 seconds: " + str(e))
                time.sleep(3)
        self.current_response = response
        self.m_episode_time = 0
        self.prevState = response.observations
        return range(0, self.num_robots), self.observation_to_list(response.observations, init=True)

    def sockService(self, actions, start_poses):
        #print('[Python] sending data request to the client')
        start_posesDict = []
        for pose in start_poses:
            start_posesDict.append([pose.x, pose.y, pose.theta])
        requestDict = {'actions': actions, 'start_poses': start_posesDict}
        reqJsonStr = json.dumps(requestDict)
        self.send(reqJsonStr)
        #####################################
        # Check that receive new observation response
        #print('[Python] waiting response data from the client')
        message = self.receive()
        while message == None:
            sleep(0.01)
            message = self.receive()
        jsonResp = json.loads(message)
        if(self.m_loop_cnt == 0):
            #print(jsonResp['observations'])
            #print(jsonResp['rewards'])
            #print(jsonResp['dones'])
            if 'params' in jsonResp:
                print(jsonResp['params'])
        ###################################
        #convert jsonResp dictionary to AI response
        response = AIServiceResponse()
        response.observations = jsonResp['observations']
        response.rewards = jsonResp['rewards']
        response.dones = jsonResp['dones']
        #because the response receive from the simulator is not in order => sorted based on ids
        #response.observations = [None]*len(jsonResp['ids'])
        #response.rewards = [None]*len(jsonResp['ids'])
        #response.dones = [None]*len(jsonResp['ids'])
        #for idx in range(len(jsonResp['ids'])):
        #    response.observations[jsonResp['ids'][idx]] = jsonResp['observations'][idx]
        #    response.rewards[jsonResp['ids'][idx]] = jsonResp['rewards'][idx]
        #    response.dones[jsonResp['ids'][idx]] = jsonResp['dones'][idx]
        #20200420 add code to get the foraging params received from simulator at the first time
        if 'params' in jsonResp:
            self.m_foraging_params = jsonResp['params']
        return response

    def send(self, message):
        message_size = str(len(message)).ljust(self.__size_message_length).encode()
        self.connection.sendall(message_size)  # Send length of msg (in known size, 16)
        self.connection.sendall(message.encode())  # Send message

    def receive(self, decode=True):
        length = self.__receive_value(self.connection, self.__size_message_length)
        if length is not None:  # If None received, no new message to read
            message = self.__receive_value(self.connection, int(length), decode)  # Get message
            return message
        return None

    def __receive_value(self, conn, buf_lentgh, decode=True):
        buf = b''
        while buf_lentgh:
            newbuf = conn.recv(buf_lentgh)
            # if not newbuf: return None  # Comment this to make it non-blocking
            buf += newbuf
            buf_lentgh -= len(newbuf)
        if decode:
            return buf.decode()
        else:
            return buf
