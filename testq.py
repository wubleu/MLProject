import gym
import numpy as np
import random
import tensorflow as tf

print()
print()
env = gym.make('Pong-v0')

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,33600],dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([33600,210],-0.1,0.1))
#W2 = tf.Variable(tf.random_uniform([210,24],-0.1,0.1))
W = tf.Variable(tf.random_uniform([210,6],-0.1,0.1))
b1 =tf.Variable(tf.zeros([210]))
#b2 = tf.Variable(tf.zeros([24]))
b = tf.Variable(tf.zeros([6]))
hidden1 = tf.nn.softmax(tf.matmul(inputs1,W1)+b1)
#hidden2 = tf.nn.sigmoid(tf.matmul(hidden1,W2)+b2)
Qout = tf.nn.softmax(tf.matmul(hidden1,W)+b)
#Qout = tf.sigmoid(tf.matmul(tf.sigmoid(tf.matmul(inputs1,W1)+b1),W)+b)
predict = tf.argmax(Qout,1)


#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,6],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
updateModel = trainer.minimize(loss)
#
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Qout,nextQ))
#updateModel = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.5
num_episodes = 20000

with tf.Session() as sess:
    sess.run(init)
    j=0
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        best = -1
        #The Q-Network
        #while j < 300:
        while True:
            env.render()
            #s_mat = s.reshape(1,100800)
            s_mat = s.reshape(-1,s.shape[-1]).astype(np.float32)
            avg = np.matrix([[0.33],[0.33],[0.33]],dtype=np.float32)
            final = np.dot(s_mat,avg)
            s_mat = np.transpose(final)
            #s_mat = s_mat/(np.max(s_mat))
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s_mat})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            rand = random.random()
            for i in range(6):
                if rand < allQ[0][i]:
                    a[0] = i
                    break
                rand-= allQ[0][i]
            
            s1,r,d,_ = env.step(a[0])
            s1_mat = s1.reshape(-1,s.shape[-1]).astype(np.float32)
            final = np.dot(s1_mat,avg)
            s1_mat = np.transpose(final)
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:s1_mat})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ.copy()
            print(allQ)
            if(r==1):
                targetQ[0,a[0]] = r + y*maxQ1
            else:
                targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W3 = sess.run([updateModel,W1],feed_dict={inputs1:s_mat,nextQ:targetQ})
            print(W3)
            s = s1
            rAll+=r
            if(r==1):
                e = 1./((i/1000) + 10)
            if r<0 or d==True:
                if(rAll>best):
                    best=rAll
                    print(best,i)
                #Reduce chance of random action as we train the model.
                e = 1./((i/1000) + 10)
                break
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")