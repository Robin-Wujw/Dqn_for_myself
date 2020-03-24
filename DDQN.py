import gym
from dueling_double_noisy_dqn import Dueling_Double_DQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = gym.make('Pendulum-v0')
env.seed(1) # 可重复实验
MEMORY_SIZE = 3000
ACTION_SPACE = 11    # 将原本的连续动作分离成 11 个动作

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = Dueling_Double_DQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False,dueling=True,noisy=False,sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = Dueling_Double_DQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True,dueling=True, sess=sess,noisy=False,output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # 在 [-2 ~ 2] 内离散化动作

        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10     # normalize 到这个区间 (-1, 0). 立起来的时候 reward = 0.
        # 立起来以后的 Q target 会变成 0, 因为 Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # 所以这个状态时的 Q 值大于 0 时, 就出现了 overestimate.

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:   # stop game
            break

        observation = observation_
        total_steps += 1
    return RL.q # 返回所有动作 Q 值

# train 两个不同的 DQN
q_natural = train(natural_DQN)
q_double = train(double_DQN)

# 出对比图
plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()