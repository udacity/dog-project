from struct import unpack
from tqdm import tqdm
import os
import glob

fileList = glob.glob('./dogImages/*/*/*.*')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

for i, image_name in enumerate(fileList):
    print(i, image_name)

    with tf.Graph().as_default():
        image_contents = tf.read_file(image_name)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        #
        with tf.Session() as sess:
            #sess.run(init_op)
            sess.run(tf.global_variables_initializer())
            tmp = sess.run(image)
