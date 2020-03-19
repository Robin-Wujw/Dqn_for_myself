import tensorflow as tf 
with tf.Session() as sess:
    saver = tf.train.Saver()
    new_saver = tf.train.import_meta_graph('model.meta')
    saver.restore(sess,'space_invaders/model.ckpt')
    var_list = [v_name for v in tf.global_variables()]
    print(var_list)
    print(sess.run(var_list))
