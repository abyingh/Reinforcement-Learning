import pygame, random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

""" AGENT """        
class DQLAgent:
    def __init__(self):
        self.state_size = 4 # distances [(p_x, e1_x), (p_y, e1_y), (p_x, e2_x), (p_y, e2_y)
        self.action_size = 3 # right, left, no move 
        
        self.gamma = 0.95
        self.learning_rate = 0.001 
        
        self.epsilon = 1  # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.model = self.nn_model()
        self.memory = deque(maxlen = 1000)
        
    def nn_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim= self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay_memory(self, batch_size):
        if len(self.memory) < batch_size:
            return
        else:
            minibatch = random.sample(self.memory, batch_size)
            
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose=0)
            
    def epsGreedy(self):
        if self.epsilon_min < self.epsilon:
            self.epsilon *= self.epsilon_decay
            
            
# window
width = 360
height = 360
fps = 60

black = (0,0,0)
white = (255,255,255)
red = (200,0,0)
green = (0,200,0)
blue = (0,0,200)


""" PLAYA """
class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        #shape
        self.image = pygame.Surface((20, 20))
        self.image.fill(red)
        self.rect = self.image.get_rect() # surround
        #initial coordinates
        self.rect.centerx = width/2       # center at 180
        self.rect.bottom = height         # bottom
        self.speedx = 4
#        self.rect.speedy = 0
        
    def update(self, action):
        self.speedx = 0
        keypress = pygame.key.get_pressed()
        
        if keypress[pygame.K_LEFT] or action == 0:
            self.speedx = -4
        elif keypress[pygame.K_RIGHT] or action == 1:
            self.speedx =  4
        else:
            self.speedx = 0
        
        self.rect.x += self.speedx
        
        # boundaries
        if self.rect.right > width:
            self.rect.right = width
        if self.rect.left < 0:
            self.rect.left = 0
            
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)
    
    
    
    
""" FOE """    
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(black)
        self.rect = self.image.get_rect()
        #initial coordinates
        self.rect.x = random.randrange(0, width - 10)
        self.rect.y = random.randrange(0, 50)
        self.speedy = 4
        self.speedx = 0
        
    def update(self):
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        
        if self.rect.top > height + 10:
            self.rect.x = random.randrange(0, width)
            self.rect.y = random.randrange(0, 50)
            self.speedy = 4
        
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)
        
    


""" ENVIRONMENT """
class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        # sprite groups & objects
        self.sprites = pygame.sprite.Group()
        self.p1 = Player()
        self.e1 = Enemy()
        self.e2 = Enemy()
        self.sprites.add(self.p1)
        self.sprites.add(self.e1)
        self.sprites.add(self.e2)
        
        self.enemy = pygame.sprite.Group()
        self.enemy.add(self.e1)
        self.enemy.add(self.e2)
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.agent = DQLAgent()
    
    
    def distance(self, a, b):
        dist = a - b
        return dist
    
    def step(self, action): # observation gets returned.
        state_list = []
        
        # update sprites
        self.p1.update(action)
        self.enemy.update()
        
        next_p1_state = self.p1.getCoordinates()
        next_e1_state = self.e1.getCoordinates()
        next_e2_state = self.e2.getCoordinates()
        
        # get returned as list, not an array!
        state_list.append(self.distance(next_p1_state[0], next_e1_state[0]))
        state_list.append(self.distance(next_p1_state[1], next_e1_state[1]))
        state_list.append(self.distance(next_p1_state[0], next_e2_state[0]))
        state_list.append(self.distance(next_p1_state[1], next_e2_state[1]))
        
        return [state_list]  # length = 1
    
    
    def initialStates(self):
        # we need to reset to get the initial state. so define all over again
        self.sprites = pygame.sprite.Group()
        self.p1 = Player()
        self.e1 = Enemy()
        self.e2 = Enemy()
        self.sprites.add(self.p1)
        self.sprites.add(self.e1)
        self.sprites.add(self.e2)
        
        self.enemy = pygame.sprite.Group()
        self.enemy.add(self.e1)
        self.enemy.add(self.e2)
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        
        
        state_list = []
        
        p1_state = self.p1.getCoordinates()
        e1_state = self.e1.getCoordinates()
        e2_state = self.e2.getCoordinates()
        
        # get returned as list, not an array!
        state_list.append(self.distance(p1_state[0], e1_state[0]))
        state_list.append(self.distance(p1_state[1], e1_state[1]))
        state_list.append(self.distance(p1_state[0], e2_state[0]))
        state_list.append(self.distance(p1_state[1], e2_state[1]))
        
        return [state_list]


    def run(self):   
         state = self.initialStates()
         running = True
         batch_Size = 24
         
         while running:
             self.reward = 1
             clock.tick(fps)
             
             for event in pygame.event.get():
                     if event.type == pygame.QUIT:
                         running = False
        
             action = self.agent.act(state)        
             next_state = self.step(action)
             self.total_reward += self.reward
             
             hits = pygame.sprite.spritecollide(self.p1, self.enemy,False, pygame.sprite.collide_circle)
             if hits:
                 self.reward = -200
                 self.total_reward += self.reward
                 running = False
                 self.done = True
                 print('total reward: {}'.format(self.total_reward))
                 
             self.agent.remember(state, action, self.reward, next_state, self.done)
            
             state = next_state
            
             self.agent.replay_memory(batch_Size)
            
             self.agent.epsGreedy()
            
            # draw & render
             screen.fill(blue)
             self.sprites.draw(screen)
             pygame.display.flip()

         pygame.quit()
        
if __name__ == '__main__':
    env = Env()
    t = 0     
    
    while True:
        t += 1        
        print('episode: {}'.format(t))
        
        pygame.init
        screen = pygame.display.set_mode((height, width))
        pygame.display.set_caption('AI agent game')
        clock = pygame.time.Clock()
        
        env.run()
