import gym
import numpy as np
import random
import tensorflow as tf

skip = 10


random_choice = True
epsilon = .01
if random_choice:
  epsilon = .7

# Set learning parameters
y = .9
e = 0.5
num_episodes = 20000
batch_size =100

#save the network
save_path='pong.save'
np.set_printoptions(threshold=np.nan)

#create weight variables
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=.01)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



#create a convolutional NN
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#max pool the convoutions
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
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
inputs1 = tf.placeholder(shape=[None,210,160],dtype=tf.float32)



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
features1 = 5
features2 = 5
aggregate = 101

W_conv1 = weight_variable([3, 3, 1, features1])
b_conv1 = bias_variable([features1])
x_image = tf.reshape(inputs1, [-1,210,160,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_pool1_flat = tf.reshape(h_pool1, [-1, 8400*features1])

W_fc1 = weight_variable([8400*features1, aggregate])
b_fc1 = bias_variable([aggregate])
keep_prob = tf.placeholder(tf.float32)

h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)


h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([aggregate, 3])
b_fc2 = bias_variable([3])


y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
Qout = tf.nn.softmax(y_conv)
#predict = tf.argmax(Qout,1)




#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[None,1,3],dtype=tf.float32)
adv = tf.placeholder(shape =[],dtype = tf.float32)
tvars = tf.trainable_variables()
one_hot = tf.placeholder(shape=[3,1],dtype=tf.float32)
loss = tf.square(nextQ - y_conv)
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
    avg = np.matrix([[0.33],[0.33],[0.33]],dtype=np.float32)    
    good_memories = []
    bad_memories = []
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        
        #convert to grey scale and normalize so the largest value is 1
        s_mat = (1/255)*s
        s_mat = (np.dot(s_mat,avg)).reshape((210,160))
        
        #we are not done yet, we just started!
        d = False
        print("Pass through",i)
        
    
        s1_mat = None
        
        #these will be the memories of the current run
        memory = []
        reward_mem = []
        while True:
            #choose a random action for the first move
            if(s1_mat==None):
              s1,r,d,_ = env.step(random.randint(0,2)+1)
              s1_mat = (np.dot(s1*(1/255),avg)).reshape((210,160))
            if(i%skip==0):
              print()
              env.render()
              
            #Choose an action by greedily (with e chance of random action) from the Q-network
            allQ,logits = sess.run([Qout,y_conv],feed_dict={inputs1:[s1_mat-s_mat], keep_prob: 1})
            a=[0]
            rand = random.random()
            if (rand < epsilon):
              if(i%skip==0):
                print("random")
              a[0]=random.randint(0,2)
              if epsilon >.01:
                epsilon = epsilon*.999
            else:
              a[0] = np.argmax(allQ[0])
            if(i%skip==0):
              print(logits)
              
            #Get new state and reward from environment
            s2,r,d,_ = env.step(a[0]+1)
            if r>0:
              reward_mem.append(r)
            s2_mat = (np.dot(s2*(1/255),avg)).reshape((210,160))
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:[s2_mat-s1_mat],keep_prob: 1})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            
            
            #Train our network using target and predicted Q values
            memory.append((s1_mat-s_mat, maxQ1,allQ.copy(), r,a))
            s_mat = s1_mat
            s1_mat = s2_mat
            
            #if we win a point or the game is over, we need to attribute rewards to our memories and train
            if not r==0 or d==True:
                memory = discount_future_reward(memory)
                if(r>0):
                  good_memories+=memory
                  if(len(good_memories)>50000):
                    good_memories = good_memories[len(memory):]
                # train
                if(r<0):
                  bad_memories+=memory
                  if(len(bad_memories)>50000):
                    bad_memories = bad_memories[len(memory):]
                total_input = []
                total_target = []
                if(len(bad_memories)>batch_size):
                  sample = random.sample(bad_memories,batch_size)
                  input_sample = []
                  target_sample = []
                  for inst in sample:
                    targetQ = inst[2].copy()
                    targetQ[0][inst[4][0]] = inst[3]+y*inst[1]
                    input_sample.append(inst[0])
                    target_sample.append(targetQ)
                  total_input+=input_sample
                  total_target+=target_sample
                if(len(good_memories)>batch_size):
                  sample = random.sample(good_memories,batch_size)
                  input_sample = []
                  target_sample = []
                  for inst in sample:
                    
                    targetQ = inst[2].copy()
                    targetQ[0][inst[4][0]] = inst[3]+y*inst[1]
                    input_sample.append(inst[0])
                    target_sample.append(targetQ)
                  total_input+=input_sample
                  total_target+=target_sample
                if(len(total_input)>0):
                  sess.run([updateModel],feed_dict={inputs1:total_input,nextQ:total_target,keep_prob: 0.5})
                memory = []
                if d==True:
                  print(reward_mem)
                  break
        if i % 5 == 0:
          saver.save(sess, save_path, global_step=i)

