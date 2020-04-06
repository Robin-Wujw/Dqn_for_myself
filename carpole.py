import gym
from dueling_double_noisy_dqn import Dueling_Double_DQN
import tensorflow as tf
env =  gym.make('CartPole-v0')
env =  env.unwrapped 

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = Dueling_Double_DQN(n_actions = env.action_space.n, 
	n_features=env.observation_space.shape[0], learning_rate=0.01,e_greedy=0.9,
	 replace_target_iter=100, memory_size=2000,e_greedy_increment=0.0008,double=True,dueling=True,noisy=False)
total_step = 0 
for episode in range(2000):
	observation = env.reset()
	ep_r = 0 
	while True:
		env.render()
		action = RL.choose_action(observation)
		observation_,reward,done,info = env.step(action)
		x, x_dot, theta, theta_dot = observation_
		r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
		r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
		reward = r1 + r2   # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
		RL.store_transition(observation,action,reward,observation_)
		if total_step>1000:
			RL.learn()
		ep_r += reward 
		if done:
			print('episode:',episode, 
				  'ep_r:',round(ep_r,2),
				  'epsilon:',round(RL.epsilon,2))
			break
		observation = observation_
		total_step += 1 
RL.plot_cost()
