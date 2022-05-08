# ML-HomeWork

## 提交办法

1. 注册 github.com 账号
2. fork 这个项目，做作业，然后提交，然后发起 pull request
3. 老师会不定时的看到你的提交，进行合并
4. 命名规范  张卓-20220507-作业1.py，请注意命名规范
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.linear_model import LogisticRegression as lr

data=pd.read_csv("fashion-mnist-demo.csv")

y,x=np.split(data,indices_or_sections=(1,),axis=1)
#分割数据
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#建立回归模型
model=lr(random_state=1)
model.fit(x_train,y_train)

#回代预测
y_pred_train=model.predict(x_train)
#测试预测
y_pred_test=model.predict(x_test)

R2=r2_score(y_test,y_pred_test)
MAE=mean_absolute_error(y_test,y_pred_test)
MSE=mean_squared_error(y_test,y_pred_test)
RMSE=sqrt(MSE)
print("R2:",R2,"MAE:",MAE,"MSE:",MSE,"RMSE:",RMSE)
