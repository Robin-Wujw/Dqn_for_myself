import numpy as np 
import pandas as pd
import tensorflow as tf
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

np.random.seed(1)
tf.set_random_seed(1)
class DoubleDQN:
    def __init__(self,
                n_actions,
                n_features,
                learning_rate=0.005,
                reward_decay=0.9,
                e_greedy=0.9, replace_target_iter=200, 
                memory_size=3000, batch_size=32,
                e_greedy_increment=None, 
                output_graph=True,
                double_q=True,
                sess = None):
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
        self.memory = np.zeros((self.memory_size,n_features*2+2))
        self.double_q = double_q
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]
        if sess is None:
            self.sess= tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess  = sess
        if output_graph:
            tf.summary.FileWriter('logs/',self.sess.graph)
        self.cost_his = []
    with tf.device('/gpu:0'):
        def _build_net(self):
            # -------------- 创建 eval 神经网络, 及时提升参数 --------------
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 用来接收 observation
            self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # 用来接收 q_target 的值, 这个之后会通过计算得到
            with tf.variable_scope('eval_net'):
                # c_names(collections_names) 是在更新 target_net 参数时会用到
                c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

                # eval_net 的第一层. collections 是在更新 target_net 参数时会用到
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

                # eval_net 的第二层. collections 是在更新 target_net 参数时会用到
                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_eval = tf.matmul(l1, w2) + b2

            with tf.variable_scope('loss'): # 求误差
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope('train'):    # 梯度下降
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

            # ---------------- 创建 target 神经网络, 提供 target Q ---------------------
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # 接收下个 observation
            with tf.variable_scope('target_net'):
                # c_names(collections_names) 是在更新 target_net 参数时会用到
                c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                # target_net 的第一层. collections 是在更新 target_net 参数时会用到
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

                # target_net 的第二层. collections 是在更新 target_net 参数时会用到
                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_next = tf.matmul(l1, w2) + b2    
    def store_transition(self,s,a,r,s_):
            if not hasattr(self,'memory_counter'):
                self.memory_counter = 0 
            transition = np.hstack((s,[a,r],s_))
            index = self.memory_counter % self.memory_size
            self.memory[index,:] = transition 
            self.memory_counter += 1
        
    def choose_action(self,observation):
        #统一observation的shape(1,size_of_observation) 统一维度:（env中是[1,size of observation])
        observation = observation[np.newaxis,:]
        actions_value = self.sess.run(self.q_eval,feed_dict={self.s:observation})
        action = np.argmax(actions_value)
        if not hasattr(self,'q'):
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99+0.01*np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0,self.n_actions)
        return action 
    def learn(self):
        #检查是否替换target_net参数 
        if self.learn_step_counter % self.replace_target_iter ==0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
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
        if self.double_q:
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