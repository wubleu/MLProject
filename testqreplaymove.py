import gym
import numpy as np
import random
import tensorflow as tf


# Set learning parameters
y = .99
e = 0.5
num_episodes = 20000
batch_size =25
save_path='pong.save'
np.set_printoptions(threshold=np.nan)
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=5e-6)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)




def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_5x5(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')

def discount_future_reward(memory_list):
  memory_list.reverse()
  cur_reward = memory_list[0][3]
  output = []
  for memory in memory_list[1:]:
    cur_reward= cur_reward*y+memory[3]
    new_mem = list(memory)
    new_mem[3] = cur_reward
    output.append(tuple(new_mem))
  return output
  

print()
print()
env = gym.make('Pong-v0')

#These lines establish the feed-forward part of the network used to choose actions
#inputs1 = tf.placeholder(shape=[None,33600],dtype=tf.float32)
inputs1 = tf.placeholder(shape=[210,160,3],dtype=tf.float32)



#W1 = tf.Variable(tf.random_uniform([33600,210],-0.1,0.1))
##W2 = tf.Variable(tf.random_uniform([210,24],-0.1,0.1))
#W = tf.Variable(tf.random_uniform([210,6],-0.1,0.1))
#b1 =tf.Variable(tf.zeros([210]))
##b2 = tf.Variable(tf.zeros([24]))
#b = tf.Variable(tf.zeros([6]))
#hidden1 = tf.nn.sigmoid(tf.matmul(inputs1,W1)+b1)
##hidden2 = tf.nn.sigmoid(tf.matmul(hidden1,W2)+b2)
#Qout = tf.nn.softmax(tf.matmul(hidden1,W)+b)
##Qout = tf.sigmoid(tf.matmul(tf.sigmoid(tf.matmul(inputs1,W1)+b1),W)+b)
#predict = tf.argmax(Qout,1)




#convolution

#features1 = 50
#features2 = 100
#aggregate = 1000
features1 = 50
features2 = 100
aggregate = 300

W_conv1 = weight_variable([6, 6, 3, features1])
b_conv1 = bias_variable([features1])
x_image = tf.reshape(inputs1, [-1,210,160,3])
h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([6, 6, features1, features2])
b_conv2 = bias_variable([features2])

h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_5x5(h_conv2)

W_fc1 = weight_variable([63*features2, aggregate])
b_fc1 = bias_variable([aggregate])
keep_prob = tf.placeholder(tf.float32)
h_pool2_flat = tf.reshape(h_pool2, [-1, 63*features2])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([aggregate, 3])
b_fc2 = bias_variable([3])


y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
Qout = tf.nn.softmax(y_conv)
#predict = tf.argmax(Qout,1)




#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,3],dtype=tf.float32)
adv = tf.placeholder(shape =[],dtype = tf.float32)
tvars = tf.trainable_variables()
#loss = tf.reduce_sum(tf.square(nextQ - Qout))*tf.neg(adv)
#loss = tf.reduce_sum(tf.mul(Qout,nextQ)*tf.neg(adv))
loss = tf.matmul(Qout,tf.transpose(nextQ))*tf.neg(adv)
#loss = tf.matmul(Qout,tf.transpose(nextQ))
#loss = tf.nn.softmax_cross_entropy_with_logits(y_conv,nextQ)
grad = tf.gradients(loss,tvars)
trainer = tf.train.AdamOptimizer(learning_rate=1e-6)


updateModel = trainer.minimize(loss)




#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, nextQ))
#updateModel = trainer.minimize(cross_entropy)
#
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Qout,nextQ))
#updateModel = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.initialize_all_variables()





with tf.Session() as sess:
    sess.run(init)
    # try load saved model
    saver = tf.train.Saver(tf.all_variables())
    load_was_success = True # yes, I'm being optimistic
    try:
      save_dir = '/'.join(save_path.split('/')[:-1])
      ckpt = tf.train.get_checkpoint_state(save_dir)
      load_path = ckpt.model_checkpoint_path
      saver.restore(sess, load_path)
    except:
      print ("no saved model to load. starting new session")
      load_was_success = False
    else:
      print ("loaded model")
      saver = tf.train.Saver(tf.all_variables())

    j=0
    good_memories = []
    bad_memories = []
    for i in range(num_episodes):
        #Reset environment and get first new observation
        avg = np.matrix([[0.33],[0.33],[0.33]],dtype=np.float32)
        s = env.reset()
        #s_mat = s.reshape(-1,s.shape[-1]).astype(np.float32)
        #final = np.dot(s_mat,avg)
        #s_mat = np.transpose(final)
        s_mat = s
        rAll = 0
        d = False
        best = -1
        print("Pass through",i)
        #The Q-Network
        #while j < 300:
        s1_mat = None
        memory = []
        while True:
            if(s1_mat==None):
              s1,r,d,_ = env.step(random.randint(0,2)+1)
              #s1_mat = s1.reshape(-1,s.shape[-1]).astype(np.float32)
              #final = np.dot(s1_mat,avg)
              #s1_mat = np.transpose(final)
              s1_mat = s1
            if(i%10==0):
              env.render()
            #Choose an action by greedily (with e chance of random action) from the Q-network
            allQ,logits = sess.run([Qout,y_conv],feed_dict={inputs1:s1_mat-s_mat, keep_prob: 1})
            a=[0]
            #if np.random.rand(1) < e:
            #    a[0] = env.action_space.sample()
            #Get new state and reward from environment
            rand = random.random()
            for j in range(3):
                if rand < allQ[0][j]:
                    a[0] = j
                    break
                rand-= allQ[0][j]
            if(i%10==0):
              print(allQ)
              print(logits)
            s2,r,d,_ = env.step(a[0]+1)
            #s2_mat = s2.reshape(-1,s.shape[-1]).astype(np.float32)
            #final = np.dot(s2_mat,avg)
            #s2_mat = np.transpose(final)
            s2_mat = s2
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:s2_mat-s1_mat,keep_prob: 1})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            #targetQ = allQ.copy()
            #if(i%1==0):
                #print(allQ,i)
            #targetQ[0,a[0]] = r + y*maxQ1     
            #Train our network using target and predicted Q values
            memory.append((s1_mat-s_mat, maxQ1,allQ.copy(), r,a))
            #_ = sess.run([updateModel],feed_dict={inputs1:s_mat,nextQ:targetQ,keep_prob: 0.5})
            s_mat = s1_mat
            s1_mat = s2_mat
            rAll+=r
            if not r==0 or d==True:
                memory = discount_future_reward(memory)
                if(r>0):
                  for mem in memory:
                      good_memories.append(mem)
                  if(len(good_memories)>50000):
                    good_memories = good_memories[len(memory):]
                # train
                if(r<0):
                  for mem in memory:
                      bad_memories.append(mem)
                  if(len(bad_memories)>50000):
                    bad_memories = bad_memories[len(memory):]
                if(len(bad_memories)>batch_size):
                  sample = random.sample(bad_memories,batch_size)
                  for inst in sample:
                    #targetQ = inst[2].copy()
                    targetQ = [[0,0,0]]
                    
                    
                    targetQ[0][inst[4][0]] = 1
                    
                    #targetQ = [[inst[1]]*3]
                    l,y_c,Q,_ = sess.run([loss,y_conv,Qout,updateModel],feed_dict={inputs1:inst[0],nextQ:targetQ,keep_prob: 0.5,adv : inst[3]})
                    #print("next")
                    #print(y_c)
                    #print(Q)
                    #print(l)
                if(len(good_memories)>batch_size):
                  sample = random.sample(good_memories,batch_size)
                  for inst in sample:
                    #targetQ = inst[2].copy()
                    targetQ = [[0,0,0]]
                    targetQ[0][inst[4][0]] = 1
                    
                    #targetQ = [[inst[1]]*3]
                    l,y_c,Q,_ = sess.run([loss,y_conv,Qout,updateModel],feed_dict={inputs1:inst[0],nextQ:targetQ,keep_prob: 0.5,adv : inst[3]})
                    #print("next")
                    #print(y_c)
                    #print(Q)
                    #print(l)
                memory = []
                if d==True:
                  break
        if i % 5 == 0:
          saver.save(sess, save_path, global_step=i)
            #print "SAVED MODEL #{}".format(episode_number)
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

