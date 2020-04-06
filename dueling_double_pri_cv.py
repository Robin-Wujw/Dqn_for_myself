import numpy as np
import tensorflow as tf
# 按顺序建立的神经网络
from keras.models import Sequential,load_model
# dense是全连接层，这里选择你要用的神经网络层参数
from keras.layers import LSTM, TimeDistributed, Dense, Activation,Convolution2D, MaxPooling2D, Flatten
# 选择优化器
from keras.optimizers import Adam, RMSprop
# 画图
from keras.utils import plot_model
np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features,
            observation_shape,
            learning_rate=1e-5,
            reward_decay=0.99,
            epsilon_max=0.95,
            replace_target_iter=300,
            memory_size=1024000,
            batch_size=32,
            e_greedy_increment=None,
            first_layer_neurno=4,
            second_layer_neurno=1,
            output_graph=False,
            dueling = True,
            double = True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.observation_shape = observation_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon_max
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.first_layer_neurno = first_layer_neurno
        self.second_layer_neurno = second_layer_neurno
        self.dueling = dueling
        self.double = double
        self.learn_step_counter = 0

        # 由于图像数据太大了 分开用numpy存
        # self.memoryList = []
        self.memoryObservationNow = np.zeros((self.memory_size, self.observation_shape[0],
                                              self.observation_shape[1], self.observation_shape[2]), dtype='int16')
        self.memoryObservationLast = np.zeros((self.memory_size, self.observation_shape[0],
                                               self.observation_shape[1], self.observation_shape[2]), dtype='int16')
        self.memoryReward = np.zeros(self.memory_size, dtype='float64')
        self.memoryAction = np.zeros(self.memory_size, dtype='int16')

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
            #print("输出图像")
            plot_model(self.model_eval, to_file='model1.png')
            plot_model(self.model_target, to_file='model2.png')

        self.cost_his = []
        self.reward = []
    def _build_net(self):
        # ------------------ 建造估计层 ------------------
        # 因为神经网络在这个地方只是用来输出不同动作对应的Q值，最后的决策是用Q表的选择来做的
        # 所以其实这里的神经网络可以看做是一个线性的，也就是通过不同的输入有不同的输出，而不是确定类别的几个输出
        # 这里我们先按照上一个例子造一个两层每层单个神经元的神经网络
        self.model_eval = Sequential([
            # 输入第一层是一个二维卷积层(100, 80, 1)
            Convolution2D(                              # 就是Conv2D层
                batch_input_shape=(None, self.observation_shape[0], self.observation_shape[1],
                                   self.observation_shape[2]),
                filters=15,                             # 多少个滤波器 卷积核的数目（即输出的维度）
                kernel_size=5,                          # 卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
                strides=1,                              # 每次滑动大小
                padding='same',                         # Padding 的方法也就是过滤后数据xy大小是否和之前的一样
                data_format='channels_last',           # 表示图像通道维的位置，这里rgb图像是最后一维表示通道
            ),
            Activation('relu'),
            # 输出(100, 80, 15)
            # Pooling layer 1 (max pooling) output shape (50, 40, 15)
            MaxPooling2D(
                pool_size=2,                            # 池化窗口大小
                strides=2,                              # 下采样因子
                padding='same',                         # Padding method
                data_format='channels_last',
            ),
            # output(50, 40, 30)
            Convolution2D(30, 5, strides=1, padding='same', data_format='channels_last'),
            Activation('relu'),
            # (10, 8, 30)
            MaxPooling2D(5, 5, 'same', data_format='channels_first'),
            # (10, 8, 30)
            Flatten(),
            # LSTM(
            #     units=1024,
            #     return_sequences=True,  # True: output at all steps. False: output as last step.
            #     stateful=True,          # True: the final state of batch1 is feed into the initial state of batch2
            # ),
            Dense(512),
            Activation('relu'),
            Dense(self.n_actions),
        ])
        # 选择rms优化器，输入学习率参数
        rmsprop = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model_eval.compile(loss='mse',
                            optimizer=rmsprop,
                            metrics=['accuracy'])

        # ------------------ 构建目标神经网络 ------------------
        # 目标神经网络的架构必须和估计神经网络一样，但是不需要计算损失函数
        self.model_target = Sequential([
            Convolution2D(  # 就是Conv2D层
                batch_input_shape=(None, self.observation_shape[0], self.observation_shape[1],
                                   self.observation_shape[2]),
                filters=15,  # 多少个滤波器 卷积核的数目（即输出的维度）
                kernel_size=5,  # 卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
                strides=1,  # 每次滑动大小
                padding='same',  # Padding 的方法也就是过滤后数据xy大小是否和之前的一样
                data_format='channels_last',  # 表示图像通道维的位置，这里rgb图像是最后一维表示通道
            ),
            Activation('relu'),
            # 输出（210， 160， 30）
            # Pooling layer 1 (max pooling) output shape (105, 80, 30)
            MaxPooling2D(
                pool_size=2,  # 池化窗口大小
                strides=2,  # 下采样因子
                padding='same',  # Padding method
                data_format='channels_last',
            ),
            # output(105, 80, 60)
            Convolution2D(30, 5, strides=1, padding='same', data_format='channels_last'),
            Activation('relu'),
            # (21, 16, 60)
            MaxPooling2D(5, 5, 'same', data_format='channels_first'),
            # 21 * 16 * 60 = 20160
            Flatten(),
            # LSTM(
            #     units=1024,
            #     return_sequences=True,  # True: output at all steps. False: output as last step.
            #     stateful=True,          # True: the final state of batch1 is feed into the initial state of batch2
            # ),
            Dense(512),
            Activation('relu'),
            Dense(self.n_actions),
        ])

        
      
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        s  = s[:,:,np.newaxis]
        s_ = s_[:,:,np.newaxis]
        index = self.memory_counter % self.memory_size
        self.memoryObservationNow[index, :] = s_
        self.memoryObservationLast[index, :] = s
        self.memoryReward[index] = r
        self.memoryAction[index] = a

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :,:,np.newaxis]
        if np.random.uniform() < self.epsilon:
            # 向前反馈，得到每一个当前状态每一个action的Q值
            # 这里使用估计网络，也就是要更新参数的网络
            # 然后选择最大值,这里的action是需要执行的action
            # print(observation)
            actions_value = self.model_eval.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
            # print(action)
        return action


    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.model_target.set_weights(self.model_eval.get_weights())
            #print('\ntarget_params_replaced\n')

        # 随机取出记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memoryONow = self.memoryObservationNow[sample_index, :]
        batch_memoryOLast = self.memoryObservationLast[sample_index, :]
        batch_memoryAction = self.memoryAction[sample_index]
        batch_memoryReward = self.memoryReward[sample_index]

        #double_Q
        q_next = self.model_target.predict(batch_memoryONow,batch_size=self.batch_size)
        q_eval_next = self.model_eval.predict(batch_memoryONow,batch_size=self.batch_size)
        
        q_eval = self.model_eval.predict(batch_memoryOLast,batch_size=self.batch_size)
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memoryAction.astype(int)
        reward = batch_memoryReward
        if self.double:
            max_act_next = np.argmax(q_eval_next,axis=1)
            selected_q_next = q_next[batch_index,max_act_next]
        else:
            selected_q_next = np.max(q_next,axis=1)
        q_target[batch_index,eval_act_index] = reward + self.gamma*selected_q_next

        self.cost = self.model_eval.train_on_batch(batch_memoryONow, q_target)

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
    def save(self):
        #保存神经网络参数
        self.model_eval.save("space_invaders_pixel/eval.h5")
        self.model_target.save("space_invaders_pixel/target.h5")
    def restore(self):
        #读取神经网络模型参数的方法
        self.model_eval = load_model('space_invaders_pixel/eval.h5')
        self.model_target = load_model('space_invaders_pixel/target.h5')
    def plot_cost(self):
        import matplotlib.pyplot as plt 
        plt.plot(np.arange(len(self.cost_his)),self.cost_his)
        plt.ylabel('Cost') 
        plt.xlabel('training steps')
        plt.show()
