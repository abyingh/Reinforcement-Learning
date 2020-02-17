import gym, random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

class DQAgent:
    def __init__(self, env):
        self.states = env.observation_space.shape[0] # 8 states
        self.actions = env.action_space.n
        
        self.alpha = 0.01
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen= 3000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim = self.states))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model
    
    def remember(self, state, action, reward, observation, done):
        self.memory.append((state, action, reward, observation, done))
        
    def action(self, state):
        if random.uniform(0,1) < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])
    
    def replay_memory(self, batch_size):
        if len(self.memory) < batch_size:
            return

#        a=np.arange(9).reshape(3,3)
#        b = np.where(a[:,1]>2)
#        (array([1, 2], dtype=int64),)
            
        # minibatch: 0-state, 1-action, 2-reward, 3-observation, 4-done
        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False) # incompleted episodes in minibatch
        y = np.copy(minibatch[:, 2])    # y = rewards of minibatch

        # If minibatch contains any non-terminal states, use separate update rule for those states        
        if len(not_done_indices[0]) > 0: # if any done = False
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 3])) # s'
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))
            
            # Non-terminal update rule
            y[not_done_indices] += np.multiply(self.gamma, predict_sprime_target[not_done_indices, np.argmax(predict_sprime[not_done_indices, :][0], axis=1)][0])

        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(batch_size), actions] = y
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=1, verbose=0)        
        
        
        
    def adaptiveEps(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        
if __name__ == '__main__':
    
    env = gym.make('LunarLander-v2')
    agent = DQAgent(env)
    
    batch_size = 16 
    episodes = 5000
    
    num_of_states = env.observation_space.shape[0]
    
    for episode in range(episodes):
        
        state = env.reset()
        state = np.reshape(state, (1, num_of_states))
        rewards = 0
        
        while True:
            env.render()
            action = agent.action(state)
            observation, reward, done, _ = env.step(action)
            observation = np.reshape(observation, (1, num_of_states))
            
            agent.remember(state, action, reward, observation, done)
        
            state = observation
            
            agent.replay_memory(batch_size)
            
            epsilon = agent.adaptiveEps()
            
            rewards += reward
            
            if done:
                print('Episode: {}, Reward: {}'.format(episode, rewards))
                break

# %% Visualization
import time
tester = agent
state = env.reset()
state = np.reshape(state, (1, 8))
counter = 0

while True:
    env.render()
    action = tester.action(state)
    observation, reward, done, _ = env.step(action)
    observation = np.reshape(observation, (1,8))
    state = observation
    counter += 1
    print(counter)
    time.sleep(0.05)
    
    if done:
        print('Done')
        break
        
 #%%       
env.close()  
