import gym
from dueling_double_noisy_dqn import Dueling_Double_DQN
import tensorflow as tf
import matplotlib.pyplot as plt 
env = gym.make('BipedalWalkerHardcore-v2')
env = env.unwrapped

print(env.action_space) 
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
ACTION_SPACE = env.action_space.shape[0]
MEMORY_SIZE = 5000
sess = tf.Session()
save_path = 'Riverraid-ram-v0/model.ckpt'
RL = Dueling_Double_DQN(
        n_actions=ACTION_SPACE, n_features=env.observation_space.shape[0], memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00002,dueling=True,double=True,noisy=False,sess=sess,output_graph=True)
total_steps = 0
RENDER = False
total_reward = 0 
i = 0 
total_reward= []
everage_reward_20 = []
try:
    RL.restore(save_path)
    print("Restore successfully")
except BaseException:
    print('No model has saved')
for i_episode in range(8000):

    observation = env.reset()
    ep_r = 0

    while True:
        if RENDER:env.render()

        action = RL.choose_action(observation)
        
        observation_, reward, done, info = env.step(np.array(action))

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
    if i%5==0:
        sum_r = total_reward[i-1]+total_reward[i-2]+total_reward[i-3]+total_reward[i-4]+total_reward[i-5]
        print("Recent 5 episodes reward:",sum_r/5)
        if sum_r/5 >=0:RENDER=True
    if i%20==0:
        everage_reward_20.append((sum(total_reward)/i))
        print("all episodes' everage reward:",sum(total_reward)/i)
    if i%300 == 0:
        print('Save successfully')
        RL.save(save_path)
    a = i/20
RL.plot_cost()
def plot_reward():
    import numpy as np
    import matplotlib.pyplot as plt 
    if i<40:
        pass
    else:
        plt.plot(np.arange(a),everage_reward_20)
        plt.ylabel('Reward') 
        plt.xlabel('training every 20 episodes')
        plt.show()
plot_reward()
