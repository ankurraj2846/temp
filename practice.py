import tensorflow as tf
a = tf.constant(1.0, name = 'a')
b = tf.Variable(0.8, name='b')

c = tf.multiply(a,b, name= 'output')

d= tf.constant(0.0, name='label')
loss = (c -d)**2
log_path = "/Users/AR/Desktop/"
optim = tf.train.GradientDescentOptimizer(learning_rate= 0.025)

g = optim.compute_gradients(loss)
i = tf.global_variables_initializer()
sess = tf.Session()
#tf.InteractiveSession() ###
## In case of function you should use g().run() since g is also a function 
sess.run(i)

k = optim.minimize(loss)

for i in range(100):
    sess.run(k)

print(sess.run(b))  

summary_y = tf.summary.scalar('output', c)
summary_writer = tf.summary.FileWriter('log_path', sess.graph)
for i in range(100):
     summary_str = sess.run(summary_y)
     summary_writer.add_summary(summary_str, i)
     sess.run(k)


#review it later python -m tf.tensorboard --logdir=log_path #tf.InteractiveSession()



