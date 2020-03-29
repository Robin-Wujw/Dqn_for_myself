import numpy as np 
import pandas as pd
import tensorflow as tf
import os 
from tensorflow.python.framework import ops
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class Dueling_Double_DQN:
	with tf.device('/gpu:0'):
		def __init__(self,
					n_actions,
					n_features,
					learning_rate=0.0005,			
					reward_decay=0.99,
					e_greedy=0.9, replace_target_iter=500, 
					memory_size=1000, batch_size=128,
					e_greedy_increment=0.00008,
					dueling = True,
					double=True,
					noisy = True,
					sess = None, 
					output_graph=True):
			self.n_actions = n_actions
			self.n_features= n_features
			self.lr = learning_rate
			self.gamma = reward_decay
			self.epsilon_max = e_greedy
			self.replace_target_iter = replace_target_iter
			self.memory_size = memory_size
			self.batch_size = batch_size
			self.epsilon_increment = e_greedy_increment
			self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
			self.learn_step_counter = 0 
			self.dueling = dueling
			self.double = double
			self.memory = np.zeros((self.memory_size,n_features*2+2))
			self.noisy = noisy
			self._build_net()
			t_params = tf.get_collection('target_net_params')
			e_params = tf.get_collection('eval_net_params')
			self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]
			if sess is None:
				self.sess  = tf.Session()
				self.sess.run(tf.global_variables_initializer())
			else:
				self.sess = sess 
			if output_graph:
				tf.summary.FileWriter('logs/',self.sess.graph)
			self.sess.run(tf.global_variables_initializer())
			self.cost_his = []
		def _build_net(self):
			n_l1, n_l2,w_initializer, b_initializer = 512,128,tf.random_normal_initializer(-0.1,0.1), tf.constant_initializer(0.1)  # config of layers			
			def noisy_dense(inputs, units, bias_shape, c_names, w_i=w_initializer, b_i=b_initializer, activation=tf.nn.relu, noisy_distribution='factorised'):
					def f(e_list):
						return tf.multiply(tf.sign(e_list), tf.pow(tf.abs(e_list), 0.5))

					if not isinstance(inputs, ops.Tensor):
						inputs = ops.convert_to_tensor(inputs, dtype='float')

					if len(inputs.shape) > 2:
						inputs = tf.contrib.layers.flatten(inputs)
					flatten_shape = inputs.shape[1]
					weights = tf.get_variable('weights', shape=[flatten_shape, units], initializer=w_i)
					w_sigma = tf.get_variable('w_sigma', [flatten_shape, units], initializer=w_i, collections=c_names)
					if noisy_distribution == 'independent':
						weights += tf.multiply(tf.random_normal(shape=w_sigma.shape), w_sigma)
					elif noisy_distribution == 'factorised':
						noise_1 = f(tf.random_normal(tf.TensorShape([flatten_shape, 1]), dtype=tf.float32))  # 注意是列向量形式，方便矩阵乘法
						noise_2 = f(tf.random_normal(tf.TensorShape([1, units]), dtype=tf.float32))
						weights += tf.multiply(noise_1 * noise_2, w_sigma)
					dense = tf.matmul(inputs, weights)
					if bias_shape is not None:
						assert bias_shape[0] == units
						biases = tf.get_variable('biases', shape=bias_shape, initializer=b_i)
						b_noise = tf.get_variable('b_noise', [1, units], initializer=b_i, collections=c_names)
						if noisy_distribution == 'independent':
							biases += tf.multiply(tf.random_normal(shape=b_noise.shape), b_noise)
						elif noisy_distribution == 'factorised':
							biases += tf.multiply(noise_2, b_noise)
						return activation(dense + biases) if activation is not None else dense + biases
					return activation(dense) if activation is not None else dense			
			def build_layers(s,c_names,n_l1,n_l2,w_initializer,b_initializer,reg=None):
				if self.noisy:
					with tf.variable_scope('l1'):
						l1 = noisy_dense(s,n_l1,[n_l1],c_names,activation=tf.nn.relu)
					with tf.variable_scope('l2'):
						l2 = noisy_dense(l1,n_l2,[n_l2],c_names,activation=tf.nn.relu)
					if self.dueling:
						with tf.variable_scope('Value'):
							self.V =noisy_dense(l2,1,[1],c_names,activation=None)
						with tf.variable_scope('Advantage'):
							self.A =noisy_dense(l2,self.n_actions,[self.n_actions],c_names,activation=None)
						with tf.variable_scope('Q'):
							#为了避免最终哦 A被学成Q(跟dqn效果一样)：当V=0时 A=Q,而A每次减去不同的值，不容易变成Q
							out = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keep_dims=True))
					else:
						with tf.variable_scope('Q'):
							out =noisy_dense(l2,self.n_actions,[self.n_actions],c_names,activation=None)
					return out
				else:
					with tf.variable_scope('l1'):
						w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
						b1 = tf.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
						l1 = tf.nn.relu(tf.matmul(s,w1)+b1)
					with tf.variable_scope('l2'):
						w2 = tf.get_variable('w2',[n_l1,n_l2],initializer=w_initializer,collections=c_names)
						b2 = tf.get_variable('b2',[1,n_l2],initializer=w_initializer,collections=c_names)
						l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)
					if self.dueling:
						with tf.variable_scope('Value'):
							w3 = tf.get_variable('w3',[n_l2,1],initializer=w_initializer,collections=c_names)
							b3 = tf.get_variable('b3',[1,1],initializer=b_initializer,collections=c_names)
							self.V = tf.matmul(l2,w3)+b3
						with tf.variable_scope('Advantage'):
							w4 = tf.get_variable('w4',[n_l2,self.n_actions],initializer=w_initializer,collections=c_names)
							b4 = tf.get_variable('b4',[1,self.n_actions],initializer=b_initializer,collections=c_names)
							self.A = tf.matmul(l2,w4)+b4
						with tf.variable_scope('Q'):
							#为了避免最终A被学成Q(跟dqn效果一样)：当V=0时 A=Q,而A每次减去不同的值，不容易变成Q
							out = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keep_dims=True))
					else:
						with tf.variable_scope('Q'):
							w5 = tf.get_variable('w5',[n_l2,self.n_actions],initializer=w_initializer,collections=c_names)
							b5 = tf.get_variable('b5',[1,self.n_actions],initializer=b_initializer,collections=c_names)
							out = tf.matmul(l2,w5)+b5
					return out 

						
			# -------------- 创建 eval 神经网络, 及时提升参数 --------------
			self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 用来接收 observation
			self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # 用来接收 q_target 的值, 这个之后会通过计算得到
			with tf.variable_scope('eval_net'):
				# c_names(collections_names) 是在更新 target_net 参数时会用到
				c_names,n_l1, n_l2,w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES],512, \
				128,tf.random_normal_initializer(-0.1,0.1), tf.constant_initializer(0.1)  # config of layers
				regularizer = tf.contrib.layers.l2_regularizer(scale=0.2)  # 注意：只有select网络有l2正则化
				self.q_eval = build_layers(self.s,c_names,n_l1,n_l2,w_initializer,b_initializer,regularizer)


			with tf.variable_scope('loss'): # 求误差
				self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
			with tf.variable_scope('train'):    # 梯度下降
				self._train_op = tf.train.RMSPropOptimizer(self.lr,0.9,0.999,1e-6).minimize(self.loss)

			# ---------------- 创建 target 神经网络, 提供 target Q ---------------------
			self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # 接收下个 observation
			with tf.variable_scope('target_net'):
				# c_names(collections_names) 是在更新 target_net 参数时会用到
				c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

				self.q_next = build_layers(self.s_,c_names,n_l1,n_l2,w_initializer,b_initializer)

		def choose_action(self,observation):
			#统一observation的shape(1,size_of_observation) 统一维度:（env中是[1,size of observation])
			observation = observation[np.newaxis,:]
			action_value = self.sess.run(self.q_eval,feed_dict={self.s:observation})
			action = np.argmax(action_value)
			if not hasattr(self,'q'):
				self.q = []
				self.running_q = 0
			self.running_q = self.running_q*0.99+0.01*np.max(action_value)
			self.q.append(self.running_q)
			if self.noisy:
				pass
			else:
				if np.random.uniform() > self.epsilon:
					action = np.random.randint(0,self.n_actions)
			return action 
		def store_transition(self,s,a,r,s_):
			if not hasattr(self,'memory_counter'):
				self.memory_counter = 0 
			transition = np.hstack((s,[a,r],s_))
			index = self.memory_counter % self.memory_size
			self.memory[index,:] = transition 
			self.memory_counter += 1

		def learn(self):
			#检查是否替换target_net参数 
			if self.learn_step_counter % self.replace_target_iter ==0:
				self.sess.run(self.replace_target_op)
				#print('\ntarget_params_replaced\n')
			#从memory中随机抽取batch_size这么多记忆 
			if self.memory_counter > self.memory_size:
				sample_index = np.random.choice(self.memory_size,size=self.batch_size)
			else:
				sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
			batch_memory = self.memory[sample_index,:]
			#(s,,[a,r],s_)
			#获取q_next(target_net产生了q)和q_eval(eval_net产生的q)
			q_next,q_eval_next = self.sess.run(
					[self.q_next,self.q_eval],
					feed_dict={
						self.s_:batch_memory[:,-self.n_features:],
						#fixed params，q_next由目标值网络用记忆库中倒数n_features个列（observation_）的值做输入
						self.s:batch_memory[:,-self.n_features:]
						# newest params，q_eval由预测值网络用记忆库中正数n_features个列（observation）的值做输入
						}
					)
			q_eval = self.sess.run(self.q_eval,{self.s:batch_memory[:,:self.n_features]})
			q_target = q_eval.copy()
			batch_index = np.arange(self.batch_size,dtype=np.int32) #[0,1,2,3....,31]
			eval_act_index = batch_memory[:,self.n_features].astype(int)
			#即RL.store_transition(observation, action, reward, observation_)中的action，注意从0开始记，所以eval_act_index得到的是action那一列
			reward = batch_memory[:,self.n_features+1] #batch_memory [s,r,a,s_]
			if self.double:
				max_act_next = np.argmax(q_eval_next,axis=1)
				selected_q_next = q_next[batch_index,max_act_next]
			else:
				selected_q_next = np.max(q_next,axis=1)
			q_target[batch_index,eval_act_index] = reward + self.gamma*selected_q_next
			#q_target 为32行2列（2个action） eval_act_index =0/1 表示两个action
			#训练eval_net 
			_,self.cost = self.sess.run([self._train_op,self.loss],
					feed_dict={self.s:batch_memory[:,:self.n_features],
						self.q_target:q_target})
			self.cost_his.append(self.cost)#记录cost误差 
			#逐渐增加epsilon 降低行为的随机性 
			self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max 
			self.learn_step_counter += 1 
			return self.cost
		def save(self,save_path):
			#保存神经网络参数
			tf.train.Saver().save(self.sess,save_path) 
		def restore(self,save_path):
			#读取神经网络模型参数的方法
			tf.train.Saver().restore(self.sess,save_path)
		def plot_cost(self):
			import matplotlib.pyplot as plt 
			plt.plot(np.arange(len(self.cost_his)),self.cost_his)
			plt.ylabel('Cost') 
			plt.xlabel('training steps')
			plt.show()
