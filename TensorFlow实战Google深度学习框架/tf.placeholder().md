# tf.placeholder函数说明

函数形式：

```python
tf.placeholder(
    dtype,
    shape=None,
    name=None
)
'''
参数：

    dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
    shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
    name：名称
'''
```

为什么要用placeholder？

```python
#   Tensorflow的设计理念称之为计算流图，在编写程序时，首先构筑整个系统的graph，代码并不会直接生效，这一点和python的其他数值计算库（如Numpy等）不同，graph为静态的，类似于docker中的镜像。然后，在实际的运行时，启动一个session，程序才会真正的运行。这样做的好处就是：避免反复地切换底层程序实际运行的上下文，tensorflow帮你优化整个系统的代码。我们知道，很多python程序的底层为C语言或者其他语言，执行一行脚本，就要切换一次，是有成本的，tensorflow通过计算流图的方式，帮你优化整个session需要执行的代码，还是很有优势的。

#   所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
```
代码示例：

```python
import tensorflow as tf
import numpy as np
 
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
 
output = tf.multiply(input1, input2)
 
with tf.Session() as sess:
    print sess.run(output, feed_dict = {input1:[3.], input2: [4.]})
```


​     
```python
import tensorflow as tf
import numpy as np
 
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)
 
with tf.Session() as sess:
    #print(sess.run(y))  # ERROR:此处x还没有赋值
    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array})) 
```