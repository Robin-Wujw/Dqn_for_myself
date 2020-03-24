import gym
from dueling_double_noisy_dqn import Dueling_Double_DQN
import tensorflow as tf
import matplotlib.pyplot as plt 
env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space) 
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
ACTION_SPACE = env.action_space.n
MEMORY_SIZE = 5000
sess = tf.Session()
save_path = 'space_invaders_noisy/model.ckpt'
RL = Dueling_Double_DQN(
        n_actions=ACTION_SPACE, n_features=env.observation_space.shape[0], memory_size=MEMORY_SIZE,
        dueling=True,double_q=True,noisy=False,sess=sess,output_graph=True)
total_steps = 0
RENDER = True
total_reward = 0 
i = 0 
total_reward= []
everage_reward_100 = []
try:
    RL.restore(save_path)
    print("Restore successfully")
except BaseException:
    print('No model has saved')
for i_episode in range(10000):

    observation = env.reset()
    ep_r = 0

    while True:
        if RENDER:env.render()

        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2   # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            if RL.noisy:
                print('Epi: ', i_episode+1,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', RL.epsilon)                
            else:    
                print('Epi: ', i_episode+1,
                      '| Ep_r: ', round(ep_r, 4),
                      '| Epsilon: ', RL.epsilon)
            break

        observation = observation_
        total_steps += 1
    total_reward.append(ep_r) 
    i= i +1
    if i>100 and i%5==0:
        sum_r = total_reward[i-1]+total_reward[i-2]+total_reward[i-3]+total_reward[i-4]+total_reward[i-5]
        print("Recent 5 episodes reward:",sum_r/5)
        if sum_r/5 >=450:RENDER=True
    if i%100==0:
        everage_reward_100.append((sum(total_reward)/i))
        print("all episodes' everage reward:",sum(total_reward)/i)
    if i%500 == 0:
        print('Save successfully')
        RL.save(save_path)
    a=i/100
RL.plot_cost()
def plot_reward():
    import numpy as np
    import matplotlib.pyplot as plt 
    plt.plot(np.arange(a),everage_reward_100)
    plt.ylabel('Reward') 
    plt.xlabel('training episode')
    plt.show()
plot_reward()
