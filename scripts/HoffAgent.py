import random
import gym
import numpy as np

STATE_WALKER = 0
STATE_BEACON = 1
PROBABILITY  = 0.3

class HoffAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        acts = []
        #print('state.shape={}'.format(state.shape))
        #for sIdx in range(0,state.shape[0]):
        for sIdx in range(0,len(state)):
            ob = state[sIdx]['obs']
            #ob[0] -> current state
            #ob[1] -> number of cummunicated beacons
            if(ob[0] == STATE_BEACON):
                if(ob[1] >= 3 and np.random.rand() <= PROBABILITY):
                    acts.append(STATE_WALKER)
                else:
                    acts.append(STATE_BEACON)
            elif(ob[0] == STATE_WALKER):
                if(ob[1] < 2):
                    acts.append(STATE_BEACON)
                else:
                    acts.append(STATE_WALKER)
            else:
                acts.append(ob[0])
        return acts  # returns action
