import gym
from RL_brain import DeepQNetwork
import tensorflow as tf
env =  gym.make('Pendulum-v0')
env.seed(1)
MEMORY_SIZE = 3000 
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
	natural_DQN = DoubleDQN(
		n_actions = ACTION_SPACE,
		n_features = 3 ,
		memory_size=MEMORY_SIZE,
		e_greedy_increment = 0.001,
		double_q = False,
		sess = sess)
with tf.variable_scope('Double_DQN'):
	double_DQN = DoubleDQN(n_actions = ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
		e_greedy_increment=0.0001,double_q=True,sess=sess,output_graph=True)
sess.run(tf.global_variables_initializer())
total_step = 0 
for episode in range(1000):
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
