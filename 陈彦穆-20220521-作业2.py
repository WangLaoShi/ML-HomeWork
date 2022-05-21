
##AdaGrad 的实现过程
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr#学习率
        self.h = None
    def update(self, params, grads):
        if self.h is None:
        	self.h = {}
        	for key, val in params.items():
            		self.h[key] = np.zeros_like(val)
    	for key in params.keys():
        	self.h[key] += grads[key] * grads[key]
        	params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            
            
##AdaDelta的实验过程
%matplotlib inline
import gluonbook as gb
from mxnet import nd
 
features, labels = gb.get_data_ch7()
 
def init_adadelta_states():
    s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    delta_w, delta_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))
 
def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * p.grad.square()
        g = ((delta + eps).sqrt() / (s + eps).sqrt()) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
        
##Adam的实现过程
%matplotlib inline
import torch
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

features, labels = d2l.get_data_ch7()

def init_adam_states():
    v_w, v_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad.data
        s[:] = beta2 * s + (1 - beta2) * p.grad.data**2
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p.data -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

'''
简单说来，岭回归就是在矩阵x"x上加一一个入I从而使得矩阵非奇异，进而能对x"x +入I求逆。
其中矩阵I是-一个mxm的单位矩阵，对角线上元素全为1,其他元素全为0。而\是-一个用户定义的数值，后面会做介绍。
在这种情况下，回归系数的计算公式将变成:
w=(xTX +λ)-'x"y
岭回归最先用来处理特征数多于样本数的情况，现在也用于在估计中加入偏差，从而得到更好的估计。这里通过引入i来限制了所有w之和,通过引入该惩罚项，能够减少不重要的参数，这个技术在统计学中也叫做缩减( shrinkage )。

加入正则项系数λ
作用：
1.增加偏差（对样本数据）# 因为直接在数据里加
减小方差（最小二乘法，预测值和真实值） # 加了正则项后，会经过最小二乘求w。
3.缩减算法： 减少无用特征的影响系数w --> 0
0*（门牌号+5） 100 eg. 缩减的作用：减少过拟合的影响
过拟合：模型对训练集的局部特征（个别特征）过分关注导致的
y = f(x) + bias + 正则项
正则项解决了1.矩阵不可求逆 2. 可能会降低噪声的影响
