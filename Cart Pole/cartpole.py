"""
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
"""

import gym ,random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQLAgent:
    def __init__(self, env):
        # parameter/hyperparameters
        self.state_size  = env.observation_space.shape[0]  # ann - input
        self.action_size = env.action_space.n              # ann - output
        
        #replay memory - training
        self.gamma = 0.95         # future reward
        self.learning_rate = 0.01 # lr
        
        # epsilone greedy
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # remember
        self.memory = deque(maxlen = 1000) # memory length
        
        self.model = self.build_model()
        
        
    def build_model(self):
        # neural network
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = "tanh"))
        model.add(Dense(self.action_size,activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model
        
    
    def remember(self, state, action, reward, observation, done):
        # storage
        self.memory.append((state, action, reward, observation, done))
    
    
    def act(self, state):
        # act with exploration or exploitation
        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
    
    
    def replay(self, batch_size): # training
        
        if len(self.memory) < batch_size:
            return  # return if there aren't enough samples
        else:
            minibatch = random.sample(self.memory, batch_size) # 
            
        for state, action, reward, observation, done in minibatch:
            if done:            # if game is over
                target = reward # 
            else:
                target = reward + self.gamma*np.amax(self.model.predict(observation)[0])
            
            train_target = self.model.predict(state)
            train_target[0][action] = target        
            self.model.fit(state, train_target, verbose= 0) # x_train, y_train
                
    def adaptive_EGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    
#%%
if __name__ == "__main__":
    
    #initialize environment & agent
    env = gym.make('CartPole-v0')
    agent = DQLAgent(env)
    
    batch_size = 16
    episodes = 50
    
    for episode in range(episodes):
        
        state = env.reset()
        state = np.reshape(state, (1,env.observation_space.shape[0])) # 4 states
        time = 0
        
        while True:
            
            #act
            action = agent.act(state)
            
            #step
            observation, reward, done, _ = env.step(action)
            observation = np.reshape(observation, (1,env.observation_space.shape[0]))
            
            #remember
            agent.remember(state, action, reward, observation, done) # 4 states, 1 action, 4 obs.
            print('state: {}, action: {}, reward: {}, observation: {}'.format(state, action,reward, observation))
            #update state
            state = observation
            
            #replay
            agent.replay(batch_size) 
            
            #adjust epsilon-greedy
            agent.adaptive_EGreedy()
            
            time += 1
            
            if done:
                print('Episode: {}, Time: {}'.format(episode, time))
                break


# test
            # max episode length = 200 (environment rule)    
            
import time
trained_model = agent      
state = env.reset() 
state = np.reshape(state, (1, env.observation_space.shape[0]))
time_t = 0

while True:
    env.render()
    action = trained_model.act(state)
    observation, reward, done, _ = env.step(action)
    observation = np.reshape(observation, (1, env.observation_space.shape[0]))
    state = observation
    time_t += 1
    print(time_t)  
    time.sleep(0.05)      
    if done:
        break
print('Done')

#%% close environment
env.close()      
            
