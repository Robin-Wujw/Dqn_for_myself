import gym
from dueling_double_priority_space import DQNPrioritizedReplay
import tensorflow as tf
import matplotlib.pyplot as plt 
env = gym.make('SpaceInvaders-ram-v4')
env = env.unwrapped
lives = env.ale.lives()
print(env.action_space) 
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
ACTION_SPACE = env.action_space.n
MEMORY_SIZE = 1024000
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)    
# sess  = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()
save_path = 'space_invaders_pri/model.ckpt'
RL = DQNPrioritizedReplay(
        n_actions=ACTION_SPACE, n_features=env.observation_space.shape[0], memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00002,sess=None,output_graph=True)
total_steps = 0
RENDER = False
i = 0
j = 0
episodes_reward= []
recent_100episodes_reward=[]
try:
    RL.restore(save_path)
    RL.epsilon = 0.95
    print("Restore successfully")
except BaseException:
    print('No model has saved')
print("Please type the number:\n 1.training 2.testing")
if input() == "1":
    TRAIN = True
else:
    TRAIN = False
for i_episode in range(13000):

    observation = env.reset()
    ep_r = 0
    if i_episode%100==0:
        episodes_reward = []
        i = 0 
    while True:
        if RENDER:env.render()

        action = RL.choose_action(observation)
        
        observation_, reward, done, info = env.step(action)
        # if lives > info['ale.lives'] and info['ale.lives'] > 0:
        #     done = True
        ep_r += reward
        if TRAIN:
            RL.store_transition(observation, action, reward, observation_)

            if (total_steps > 1000) and (total_steps%5==0):
                RL.learn()
        else:
            RENDER =True
            RL.epsilon=1
        if done:
            # print('Epi: ', i_episode+1,
            #       '| Ep_r: ', round(ep_r, 4),
            #       '| Epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1
    episodes_reward.append(ep_r)
    i= i + 1
    j= j + 1
    if i%10 ==0:
        sum_r = sum(episodes_reward[i-10:i])
        print("Recent 10 episodes reward:",round(sum_r/10,10))
        if sum_r/10 >=570:RENDER=True
    if j%100==0:
        recent_100episodes_reward.append((sum(episodes_reward)/i))
        print("Recent 100 episodes' reward:",round(sum(episodes_reward)/100,5))

    if j%300 == 0:
        print('Save successfully')
        RL.save(save_path)
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
