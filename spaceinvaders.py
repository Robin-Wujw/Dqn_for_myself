import gym
from dueling_double_pri_cv import DQNPrioritizedReplay
import cv2
import numpy as np
import time
from receiveThread import myThread
import tensorflow as tf
env = gym.make('SpaceInvaders-v0')
env = env.unwrapped

print(env.action_space)
# print(env.observation_space)
print(env.observation_space.shape)
print(env.observation_space.high)
print(env.observation_space.low)
print(env.reward_range)

inputImageSize = (84, 84, 1)
# inputImageSize[2] = 1
RL = DQNPrioritizedReplay(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  observation_shape=inputImageSize,
                  learning_rate=0.00025, epsilon_max=0.9,
                  replace_target_iter=300, memory_size=102400,
                  e_greedy_increment=0.0001,
                  output_graph=True)
save_path = 'space_invaders_pixel'
total_steps = 0
RENDER = False
i = 0
j = 0
episodes_reward= []
recent_100episodes_reward=[]
try:
    RL.restore()
    RL.epsilon = 0.9
    print("Restore successfully")
except BaseException:
    print('No model has saved')
print("Please type the number:\n 1.training 2.testing")
if input() == "1":
    TRAIN = True
else:
    TRAIN = False
for i_episode in range(6000):

    observation = env.reset()
    # 使用opencv做灰度化处理
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(observation, (inputImageSize[1], inputImageSize[0]))
    ep_r = 0
    if i_episode%100==0:
        episodes_reward = []
        i = 0 
    while True:
        if RENDER:env.render()
        # observation_, reward, done, info = env.step(env.action_space.sample())
        # print(env.action_space.sample())
        # # observation_, reward, done, info = env.step(4)  # 4是发送子弹 2、3分别是左右
        # if reward > 0:
        #     print(reward)
        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        # 给reward做归一化处理
        # 使用opencv做灰度化处理
        observation_ = cv2.cvtColor(observation_, cv2.COLOR_BGR2GRAY)
        observation_ = cv2.resize(observation_, (inputImageSize[1], inputImageSize[0]))
        # cv2.imshow('obe', observation_)
        ep_r += reward

        if TRAIN:
            RL.store_transition(observation, action, reward, observation_)

            if (total_steps > 1000) and (total_steps%5==0):
                t0 = time.time()
                RL.learn()
                t1 = time.time()
                if total_steps < 1010:
                    print("学习一次时间：", t1 - t0)

        else:
            RENDER =True
            RL.epsilon=1
        if done:
            print('Epi: ', i_episode+1,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1
    episodes_reward.append(ep_r)
    i= i + 1
    j= j + 1
    if i%10 ==0:
        sum_r = sum(episodes_reward[i-10:i])
        print("Recent 10 episodes reward:",round(sum_r/10,10))
        if sum_r/10 >=470:RENDER=True
    if j%100==0:
        recent_100episodes_reward.append((sum(episodes_reward)/i))
        print("Recent 100 episodes' reward:",round(sum(episodes_reward)/100,5))

    if j%200 == 0:
        print('Save successfully')
        RL.save()
    a = j/100    
RL.plot_cost()
def plot_reward():
    import numpy as np
    import matplotlib.pyplot as plt 
    if i<40:
        pass
    else:
        plt.plot(np.arange(a),  recent_100episodes_reward,5)
        plt.ylabel('Reward') 
        plt.xlabel('training every 100 episodes')
        plt.show()
plot_reward()