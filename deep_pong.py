import gym
import numpy as np
import random
import tensorflow as tf
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

#how often do we want to render?
skip = 50

#the starting learn rate
learn_rate = 1e-7

#the minimum learn rate
learn_rate_min = 1e-8

#the decay of the learn rate
learn_decay = .999

#the decay of the randomness
epsilon_decay = .999992
#do we want to start with randomness? (for testing purposes)
random_choice = True
epsilon = .01
if random_choice:
  epsilon = 1

# Set learning parameters
#we use two different discounted future rewards
y = .99
y2 = .97
num_episodes = 20000
batch_size =200

#save the network
save_path='pong.save'
np.set_printoptions(threshold=np.nan)

#create weight variables
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(.3, shape=shape)
  return tf.Variable(initial)


#we don't end up using these
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

#because we know how pong works, we can assign a reward to every frame after the game is done
def discount_future_reward(memory_list):
  memory_list.reverse()
  cur_reward = memory_list[0][3]
  output = []
  for memory in memory_list[1:]:
    cur_reward= cur_reward*y2+memory[3]
    new_mem = list(memory)
    new_mem[3] = cur_reward
    output.append(tuple(new_mem))
  return output
  

print()
print()
env = gym.make('Pong-v0')



#grey scale and crop and flatten the image
def preprocess(image):
  avg = np.array([[0.33],[0.33],[0.33]],dtype=np.float32)   
  image = np.dot(image,avg)*(1/255)
  image1 = image[35:195,15:145]
  image2 = image1.ravel()
  return image2


#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[None,20800],dtype=tf.float32)

#number of hidden nodes in each of the two layers
hidden1 = 100
hidden2 = 50

#we tried dropout at one point
keep_prob = tf.placeholder(tf.float32)

W1 = weight_variable([20800,hidden1])
b1 = bias_variable([hidden1])

layer1 = tf.nn.relu(tf.matmul(inputs1,W1)+b1)
W2 = weight_variable([hidden1,hidden2])
b2 = bias_variable([hidden2])

layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
W3 = weight_variable([hidden2, 2])
#b3 = bias_variable([2])

Qout = tf.matmul(layer2,W3)



#y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#Qout = tf.nn.softmax(y_conv)
#predict = tf.argmax(Qout,1)




#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
#we matrix multiply by a one-hot vector of the action taken so that we only change inputs for that action
nextQ = tf.placeholder(shape=[None,1,2],dtype=tf.float32)
#adv = tf.placeholder(shape =[],dtype = tf.float32)
#tvars = tf.trainable_variables()
one_hot = tf.placeholder(shape=[None,2,1],dtype=tf.float32)

loss = tf.batch_matmul(tf.square(nextQ - Qout),one_hot)
#loss = tf.square(nextQ-Qout)
#grad = tf.gradients(loss,tvars)
trainer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)

updateModel = trainer.minimize(loss)




#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, nextQ))
#updateModel = trainer.minimize(cross_entropy)
#
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Qout,nextQ))
#updateModel = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



init = tf.initialize_all_variables()

#we will be training on both good and bad memories (pos and neg reward)
good_ind = 0
bad_ind = 0
all_rewards = []
memory_size = 40000
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

    #we will keep the memories in these lists
    bad_memories = []
    bad_target = []
    bad_onehot = []
    good_memories = []
    good_target = []
    good_onehot = []
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        
        #convert to grey scale and normalize so the largest value is 1
        s_mat =preprocess(s)
        
        
        #we are not done yet, we just started!
        d = False
        print("Pass through",i)
        
        print(learn_rate)
        s1_mat = None
        
        #these will be the memories of the current run
        memory = []
        reward_mem = []
        while True:
            #choose a random action for the first move
            if(s1_mat==None):
              s1,r,d,_ = env.step(random.randint(0,1)+2)
              
              s1_mat = preprocess(s1)
              
            if(i%skip==0):
              print()
              env.render()
    
            #Choose an action greedily (with epsilon chance of random action) from the Q-network
            allQ = sess.run(Qout,feed_dict={inputs1:[s1_mat-s_mat],keep_prob:1.0})
            a=[0]
            rand = random.random()
            if (rand < epsilon):
              if(i%skip==0):
                print("random")
              a[0]=random.randint(0,1)
              if epsilon >.01:
                epsilon = epsilon*epsilon_decay
            else:
              a[0] = np.argmax(allQ[0])
            if(i%skip==0):
              print(allQ[0])
              
            #Get new state and reward from environment
            s2,r,d,_ = env.step(a[0]+2)
            if r>0:
              reward_mem.append(r)
            
            s2_mat = preprocess(s2)
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:[s2_mat-s1_mat],keep_prob:1.0})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            
            
            #store this memory so we can compute the discounted future reward when we are done
            memory.append((s1_mat-s_mat, maxQ1,allQ.copy(), r,a))
            s_mat = s1_mat
            s1_mat = s2_mat
            #if j > 30:
            #  imgplot = plt.imshow(s2_mat[35:195,15:145],cmap="gray")
            #  plt.show()

            #if we win a point or the game is over, we need to attribute rewards to our memories and train
            if not r==0 or d==True: 
                memory = discount_future_reward(memory)
                #figure out if it was a good or bad memory
                if r>0:
                  all_memories = good_memories
                  all_target = good_target
                  all_onehot = good_onehot
                  ind = good_ind
                else:
                  all_memories = bad_memories
                  all_target = bad_target
                  all_onehot = bad_onehot
                  ind = bad_ind
                #add the new memory, overwriting old memories if necessary
                for k,mem in enumerate(memory):
                  if len(all_memories)<memory_size:
                    all_memories.append(0)
                    all_target.append(0)
                    all_onehot.append(0)
                  all_memories[ind] = mem[0]
                  one = [[0],[0]]
                  one[mem[4][0]] = [1]
                  targetQ = mem[2].copy()
                  if k<10:
                    targetQ[0][mem[4][0]] = mem[3]
                  else:
                    targetQ[0][mem[4][0]] = mem[3] + y*mem[1]
                  all_target[ind] = targetQ
                  all_onehot[ind]=one
                  ind = (ind+1)%memory_size
                if r> 0:
                  good_ind = ind
                  good_range = 2
                  bad_range = 1
                else:
                  bad_range = 2
                  good_range = 1
                  bad_ind = ind
                # train
                #to try to get out of local mins, we train on both good and bad memories
                #but train on twice as many as the type (good or bad) of memory that just happened
                cur_batch = int(len(all_memories)*batch_size/memory_size)
                if(len(good_memories)>cur_batch and cur_batch>0):
                  for train_passes in range(good_range):
                    sample = random.sample(range(len(good_memories)),cur_batch)
                    input_sample = [good_memories[j] for j in sample]
                    target_sample = [good_target[j] for j in sample]
                    onehot_sample = [good_onehot[j] for j in sample]
                    sess.run([updateModel],feed_dict={inputs1:input_sample,nextQ:target_sample,one_hot:onehot_sample,keep_prob:0.7})
                if(len(bad_memories)>cur_batch and cur_batch>0):
                  for train_passes in range(bad_range):
                    sample = random.sample(range(len(bad_memories)),cur_batch)
                    input_sample = [bad_memories[j] for j in sample]
                    target_sample = [bad_target[j] for j in sample]
                    onehot_sample = [bad_onehot[j] for j in sample]
                    sess.run([updateModel],feed_dict={inputs1:input_sample,nextQ:target_sample,one_hot:onehot_sample,keep_prob:0.7})
                if learn_rate > learn_rate_min :
                  learn_rate=learn_rate*learn_decay
                memory = []
                if d==True:
                  print(reward_mem)
                  all_rewards.append(len(reward_mem))
                  break
        #save the variables to a file
        if i % 5 == 0:
          saver.save(sess, save_path, global_step=i)
        #write the number of games won in each episode to a file
        if i %100 == 0:
          f = open("output_"+str(i), 'w')
          for j in all_rewards:
            f.write(str(j)+"\n")
          f.close()


