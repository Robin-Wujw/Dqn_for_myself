import gym
from dueling_double_dqn import Dueling_Double_DQN
import tensorflow as tf
import matplotlib.pyplot as plt 
env = gym.make('SpaceInvaders-ram-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
ACTION_SPACE = env.action_space.n
MEMORY_SIZE = 5000
sess = tf.Session()
save_path = 'space_invaders/model.ckpt'
RL = Dueling_Double_DQN(
        n_actions=ACTION_SPACE, n_features=env.observation_space.shape[0], memory_size=MEMORY_SIZE,
        e_greedy_increment=None,dueling=True,double_q=True,sess=sess,output_graph=True)
total_steps = 0
RENDER = False
total_reward = 0 
i = 0 
total_reward= []
try:
    RL.restore(save_path)
except BaseException:
    print('No model has saved')
for i_episode in range(30000):

    observation = env.reset()
    ep_r = 0

    while True:
        if RENDER:env.render()

        action = RL.choose_action(observation)
        
        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            print('Epi: ', i_episode+1,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1
    total_reward.append(ep_r) 
    i= i +1
    if i>100 and i%5==0:
        sum_r = total_reward[i-1]+total_reward[i-2]+total_reward[i-3]+total_reward[i-4]+total_reward[i-5]
        print(sum_r/5)
        if sum_r/5 >=600:RENDER=True
    if i%1000 == 0:
        print('save')
        RL.save(save_path)
RL.plot_cost()
def plot_reward():
    import numpy as np
    import matplotlib.pyplot as plt 
    plt.plot(np.arange(len(total_reward)),total_reward)
    plt.ylabel('Reward') 
    plt.xlabel('training episode')
    plt.show()
plot_reward()
