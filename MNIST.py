'''
@title:Understand MNIST by Neil_jinsh_public

@author:Neil_jinsh_public

@Date:2019-9-23
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
在TensorFlow中导入Input_data数据
Input_data数据导入命名为Mnist并规定label格式为one_hot
'''


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # keep_prob: A scalar Tensor with the same type as x. The probability that each element is kept.
    # 意思是每个元素被保留的概率，那么    keep_prob: 1    就是所有元素全部保留的意思。

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # equal返回内部参数是否相等的判断，相等返回True 否则返回Flase

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 将correct_prediction得到的值转换为float32类型数值，并计算矩阵平均值

    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


'''

定义准确度计算函数
需要用到的输入参数v_xs以及v_ys（输入的测试集图片以及相对应的label标签）
首先'global perdiction'为查找全局变量perdiction，在计算y_pre时使用到prediction
feed_dict为字典，存储着名为xs和ys的图片数据以及label数据

'''


def weight_variable(shape):  # 定义w权值矩阵，shape为矩阵大小
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # 定义b，shape为矩阵大小
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


'''
方法定义
    tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

参数：
    input : 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
    filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
    strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
    padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
    use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true
'''


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # ksize  [1,pool_op_length,pool_op_width,1]
    # Must have ksize[0] = ksize[3] = 1
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


'''
方法定义
    tf.nn.max_pool(value, ksize, strides, padding, name=None)

参数：
    参数是四个，和卷积很类似：
    第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
    第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'

    返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
'''

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])  # 10x 1
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # 输入input图片灰度值范围[-1,1]，28*28
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable(
    [5, 5, 1, 32])  # patch 5x5, in size 1, out size 32  【权重为5*5矩阵，输入1张图片，输出32个feature map，即存在32种filter】
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一层卷积 output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # 第一层池化 output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

##flat h_pool2##
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]

## fc1 layer ##     # 全连接层1
W_fc1 = weight_variable([7 * 7 * 64, 1024])  # input size 7*7*64 , output size 1024
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 此处keep_prob = 0.5

'''
方法定义：
    tf.nn.dropout(    x,    keep_prob,    noise_shape=None,     seed=None,    name=None)

参数：
    x：指输入，输入tensor
    keep_prob: float类型，每个元素被保留下来的概率，设置神经元被选中的概率,在初始化时keep_prob是一个占位符, keep_prob = tf.placeholder(tf.float32) 。tensorflow在run时设置keep_prob具体的值，例如keep_prob: 0.5
    noise_shape  : 一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape。
    seed : 整形变量，随机数种子。
    name：指定该操作的名字

dropout必须设置概率keep_prob，并且keep_prob也是一个占位符，跟输入是一样的
keep_prob = tf.placeholder(tf.float32)
train的时候才是dropout起作用的时候，test的时候不应该让dropout起作用
'''

## fc2 layer ##     # 全连接层2
W_fc2 = weight_variable([1024, 10])  # input size 1024 , output size 10
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss function
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 这里用到了AdamOptimizer优化器，由于这个计算量很大，用GradientDescentOptimizer优化器下降速度太慢，所以用AdamOptimizer

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))