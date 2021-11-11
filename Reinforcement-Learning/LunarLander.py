import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQLAgent:
    def __init__(self, env):
        # parameters/hyperparameters
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.gamma = 0.95
        self.learning_rate = 0.001

        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=1000)

        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        # neural network model for deep q learning
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # acting: explore or exploit
        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])

        if len(not_done_indices[0]) > 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))

            y[not_done_indices] += np.multiply(self.gamma, predict_sprime_target[
                not_done_indices, np.argmax(predict_sprime[not_done_indices, :][0], axis=1)][0])
        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(batch_size), actions] = y
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=1, verbose=0)

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())


if __name__ == "__main__":

    # initialize env and agent
    env = gym.make("LunarLander-v2")
    agent = DQLAgent(env)
    batch_size = 32
    episodes = 10
    for e in range(episodes):

        # initialize environment
        state = env.reset()

        state = np.reshape(state, [1, 8])

        total_reward = 0
        for time in range(1000):

            # act
            action = agent.act(state)

            # step
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 8])
            # remember
            agent.remember(state, action, reward, next_state, done)
            # update state
            state = next_state
            # replay
            agent.replay(batch_size)

            total_reward += reward

            if done:
                agent.targetModelUpdate()
                break

        # epsilon decay
        agent.adaptiveEGreedy()

        # Running average of past 100 episodes
        print('Episode: {}, Reward: {}'.format(e, total_reward))

    # %% test
import time

trained_model = agent
state = env.reset()
state = np.reshape(state, [1, env.observation_space.shape[0]])
while True:
    env.render()
    action = trained_model.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
    state = next_state
    time.sleep(0.4)
    if done:
        break
print("Done")










