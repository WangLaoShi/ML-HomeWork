# Ridge算法

文章来源：CSDN博主「易长安」
原文链接：https://blog.csdn.net/qq_43019258/article/details/108939783

## 1. Ridge模型

Ridge也是线性回归的一个拓展，其模型与线性回归模型一样，最终都是得到一个线性方程：

![image-20220517174345867](http://rfm.oss-cn-beijing.aliyuncs.com/img/image-20220517174345867.png)

其中在训练阶段f ( x ) 和 x f(x)和xf(x)和x都是已知的，w 和 b w和bw和b是需要估计的参数。

## 2. Ridge策略

Rigde是在线性回归的损失函数的基础上，加入了L2惩罚项，`可以解决共线性问题`。
Ridge损失函数：

![image-20220517174621606](http://rfm.oss-cn-beijing.aliyuncs.com/img/image-20220517174621606.png)

![image-20220517174749429](http://rfm.oss-cn-beijing.aliyuncs.com/img/image-20220517174749429.png)

红色椭圆形就是L2范数的约束，最终最优解不一定落在顶点处。所以L2不具备变量筛选的功能，但是为什么可以解决共线性问题呢？

## 3. Ridge算法

上一小节我们推导出线性回归参数的最优解

![image-20220517174835200](http://rfm.oss-cn-beijing.aliyuncs.com/img/image-20220517174835200.png)

![image-20220517175209148](http://rfm.oss-cn-beijing.aliyuncs.com/img/image-20220517175209148.png)
