# tensorflow中 tf.reduce_mean函数

tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。

  

```python
reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None)
```

 


```python
'''
第一个参数input_tensor： 输入的待降维的tensor;
第二个参数axis： 指定的轴，如果不指定，则计算所有元素的均值;
第三个参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
第四个参数name： 操作的名称;
第五个参数 reduction_indices：在以前版本中用来指定轴，已弃用;
'''
```

 


以一个维度是2，形状是[2,3]的tensor举例：

```python
import tensorflow as tf
 
x = [[1,2,3],
      [1,2,3]]
 
xx = tf.cast(x,tf.float32)
 
mean_all = tf.reduce_mean(xx, keep_dims=False)
mean_0 = tf.reduce_mean(xx, axis=0, keep_dims=False)
mean_1 = tf.reduce_mean(xx, axis=1, keep_dims=False)
```


​     
```python
with tf.Session() as sess:
    m_a,m_0,m_1 = sess.run([mean_all, mean_0, mean_1])
 
print m_a    # output: 2.0
print m_0    # output: [ 1.  2.  3.]
print m_1    #output:  [ 2.  2.]
```

如果设置保持原来的张量的维度，keep_dims=True ，结果：

```python
print m_a    # output: [[ 2.]]
print m_0    # output: [[ 1.  2.  3.]]
print m_1    #output:  [[ 2.], [ 2.]]
```


  

类似函数还有:

    tf.reduce_sum ：计算tensor指定轴方向上的所有元素的累加和;
    tf.reduce_max  :  计算tensor指定轴方向上的各个元素的最大值;
    tf.reduce_all :  计算tensor指定轴方向上的各个元素的逻辑和（and运算）;
    tf.reduce_any:  计算tensor指定轴方向上的各个元素的逻辑或（or运算）;