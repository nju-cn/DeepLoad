import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd

def generate():
    a = {3:[[0.5,0.3,0.2],[0.6,0.2,0.2],[0.7,0.2,0.1],[0.8,0.1,0.1]],
             2:[[0.5,0.5,0],[0.6,0.4,0],[0.7,0.3,0],[0.8,0.2,0],[0.9,0.1,0]],
             1:[[1,0,0]]}
    result =[]
    list_=[[0,3,4],[0,4,3],[0,4,5],[0,5,4],[0,3,5],[0,5,3],[0,-1,-1],
           [1,6,7],[1,7,6],[1,7,8],[1,8,7],[1,6,8],[1,8,6],[1,-1,-1],
           [2,9,10],[2,10,9],[2,9,11],[2,11,9],[2,10,11],[2,11,10],[2,-1,-1]]
    for item in list_:
        if item[1]==-1 and item[2]==-1:
            for item_1 in a[1]:
                result.append(item + item_1)
        if item[0]!=-1 and item[1]!=-1 and item[2]!=-1:
            for item_1 in a[3]:
                result.append((item+item_1))
    return result

ACTION = generate()
r_ul_max=175
r_dl_max=275
r_ul_min=125
r_dl_min=225
cpu = [4,4,4,5,5,6,4,4,5,3,5,4]
class Offloading():
    def ___init__(self):
        self.action_space = list(range(len(ACTION)))
        self.state = None
        self.application = None
    def step(self,action):
        state = self.state
        b_ul,b_dl,b_p,queue=state[0:3],state[3:6],state[6:15],state[15:]
        W, B, ddl = self.application[0],self.application[1],self.application[2]
        # 将动作分开
        target_action = ACTION[action]
        target_AP, target_ae1,target_ae2, P_0, P_ae1, P_ae2 = target_action[0], target_action[1],target_action[2], target_action[3], target_action[4], target_action[5]
        
        #全部都在本地执行
        #贪心算法——：每次选择链路最优的：
        



        #定义奖赏函数
        if P_0==1:
            D_total = B/b_ul[target_AP]+0.6*B/b_dl[target_AP]+(queue[target_AP]+W)/cpu[target_AP]
            reward = (ddl-50-D_total)/float(D_total)
            #reward = (ddl-D_total-10)*1.5
            if ddl-50>D_total:
             #   reward = 20
                satisfied=1
            else:
              #  reward= -20
                satisfied=0
            #reward = -D_total/float(10000)
        else:
            D_tr = B/b_ul[target_AP]+0.6*B/b_dl[target_AP]
            D_total = D_tr+max((P_0*W+queue[target_AP])/cpu[target_AP],b_p[target_ae1-4]+(queue[target_ae1]+W*P_ae1)/cpu[target_ae1],b_p[target_ae2-4]+(queue[target_ae2]+W*P_ae2)/cpu[target_ae2])
            reward = (ddl-50-D_total)/float(ddl-50)
            #reward = -1.2*D_total/float(10000)
            #reward = (ddl-D_total)-10
            if ddl-50>D_total:
            #    reward = 15
                satisfied=1
            else:
             #   reward= -40
                satisfied=0

        #定义新环境
        
        x = np.random.uniform()
        if (x>0.6):
            b_ul[target_AP] =min(r_ul_max,b_ul[target_AP]-1)
            b_dl[target_AP] =min(r_dl_max,b_dl[target_AP]-1)
        else:
            b_ul[target_AP] = max(r_ul_min, b_ul[target_AP] +2)
            b_dl[target_AP] = max(r_dl_min, b_dl[target_AP] +2)
        for i in range(len(b_dl)):
            if i!=target_AP:
                temp1 = b_ul[i] + np.random.randint(-2,2)
                temp2 = b_dl[i] + np.random.randint(-3,3)
                if temp1>r_ul_min and temp1<r_ul_max:
                    b_ul[i] = temp1
                if temp2>r_dl_min and temp2<r_dl_max:
                    b_dl[i] = temp2
        # for i in range(len(b_p)):
        #     if i!=target_ae1-4 and i!=target_ae2-4:
        #         b_p[i] += np.random.randint(-2,2)
        #     else:
        #         b_p[i] += np.random.randint(-1,3)
        for i in range(len(queue)):
            if i ==target_AP or i==target_ae1 or i==target_ae2:
                queue[i] += np.random.randint(-2,4)
            else:
                queue[i] += np.random.randint(-4,4)


        new_state = np.concatenate((b_ul,b_dl,b_p,queue))
        self.state = new_state
        return self.state,reward,satisfied

    def reset(self):
        b_ur = np.random.randint(125, 175, 3)
        b_dl = np.random.randint(225, 275, 3)
        b_p = np.random.randint(25, 35, 9)
        queue = np.random.randint(125, 175, 12)
        # 生成一个任务
        app = self.reset_app()
        state = np.concatenate((b_ur, b_dl, b_p, queue))
        self.state = state
        return state

    def reset_app(self):
#        W = np.random.randint(300, 400)
 #       B = np.random.randint(34000, 36000)
  #      
        W = 500
        B = 40000
        ddl = B/150 + 0.6 * B/250 + (150 + W)/4
        ddl = int(ddl)
        self.application = np.array([W,B,ddl])
        return self.application


OUTPUT_GRAPH = True
LOG_DIR = './log'
#N_WORKERS = multiprocessing.cpu_count()
N_WORKERS = 30
file_name = input('请输入文件名：')
MAX_GLOBAL_EP = int(input('请输入训练次数：'))
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
SATISFIED_RATE = []
GLOBAL_EP = 0
MAX_COUNT = 100


env = Offloading()
N_S = 27
N_A = len(ACTION)


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td)) # critic的loss是平方loss

                with tf.name_scope('a_loss'):
                    # Q * log（
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) *
                                             tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                                             axis=1, keepdims=True)
                    exp_v = log_prob * tf.stop_gradient(td) # 这里的td不再求导，当作是常数
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keepdims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'): # 把主网络的参数赋予各子网络
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'): # 使用子网络的梯度对主网络参数进行更新
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a_1 = tf.layers.dense(l_a, 200, tf.nn.softmax, kernel_initializer=w_init, name='la1') # 得到每个动作的选择概率
            a_prob = tf.layers.dense(l_a_1, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap') # 得到每个动作的选择概率
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_c_1 = tf.layers.dense(l_c, 100, tf.nn.softmax, kernel_initializer=w_init, name='lc1') # 得到每个动作的选择概率
            v = tf.layers.dense(l_c_1, 1, kernel_initializer=w_init, name='v')  # 得到每个状态的价值函数
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = env
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            count=0
            while True:
                a = self.AC.choose_action(s)
                s_, r, sat= self.env.step(a)
                ep_r += r
                if sat == 1:
                    count += 1
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                if total_step % UPDATE_GLOBAL_ITER == 0 :   # update global and assign to local net
                    v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_ # 使用v(s) = r + v(s+1)计算target_v
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if total_step%30==0:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    SATISFIED_RATE.append(float(count/30))
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],"|satisfied:%.2f"%(float(count/30))
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()
    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    # Coordinator类用来管理在Session中的多个线程，
    # 使用 tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job) # 创建一个线程，并分配其工作
        t.start() # 开启线程
        worker_threads.append(t)
    COORD.join(worker_threads) #把开启的线程加入主线程，等待threads结束

    res = np.concatenate([np.arange(len(GLOBAL_RUNNING_R)).reshape(-1,1),np.array(GLOBAL_RUNNING_R).reshape(-1,1),np.array(SATISFIED_RATE).reshape(-1,1)],axis=1)
    pd.DataFrame(res, columns=['episode', 'a3c_reward','stisfied rate']).to_csv(file_name)

    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('step')
    # plt.ylabel('Total moving reward')
    # plt.show()

  
    
    
