import numpy as np
from sklearn import linear_model,metrics


def ridgeRegres(xMat, yMat, lam=0.2):
    # xMat = np.mat(xMat);yMat = np.mat(yMat)
    xTx = xMat.T * xMat

    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:  # lam为零时，还是有可能为奇异阵的
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest( xArr, yArr):
    # xArr =np.array(xArr) ;yArr =np.array(yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)  # 数据标准化 可以使用regularize()函数
    # yMean = np.mean(yMat,0)
    # yMat = yMat-yMean
    # xMeans = np.mean(xMat,0)
    # xVar = np.var(xMat,0)
    # xMat = (xMat-xMeans)/xVar
    numTestPts = 22  # 设置岭参数lam的取值范围
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T  # 每一行是一个lam对应的回归系数

    return wMat


def ols(x, y):
    '''
    函数说明：最小二乘法拟合直线
    :param x: 特征值列表或者数组
    :param y: 目标值列表或者数组
    :return: 均方根误差（标准误差）和相关系数R^2（函数计算和手动计算）
             拟合图形
    '''
    # x = np.array(x); y = np.array(y)

    # self.coef = np.linalg.lstsq(x, y)[0]     #系数矩阵

    coef = np.array(ridgeTest(x, y))

    #fig, ax = plt.subplots()
    #ax.plot(self.coef)
    # plt.show()
    # self.coef = self.coef.reshape((self.Preprocess.onehot.shape[1],))       #将（x,1）变成（x，）

    coef = coef[14]#调试而得的14
    yPred = []
    y = np.array(y)
    for i in range(len(x)):
        yPred.append(np.dot(coef, x[i]))  # 将的到得系数矩阵和x相乘再相加，得到y的预测值
    yPred = np.array(yPred)  # 将预测值变成数组
    y = y.reshape((len(y),))  # 变形成一维数组
    sub1 = (y - yPred) * (y - yPred)  # 求差值再平方
    MSE = sub1.sum() / len(x)

    #print('手动RMSE:', np.sqrt(MSE))
    #print('函数RMSE:', np.sqrt(metrics.mean_squared_error(y, yPred)))

    # print(pd.DataFrame({'a':yPred}).a.sort_values())

    yavg = np.array([y.mean()] * len(y))
    sub2 = (y - yavg) * (y - yavg)
    R2 = 1 - sub1.sum() / sub2.sum()

    #print('手动R2:', R2)
    #print('函数R2:', metrics.r2_score(y, yPred))


    '''ax.scatter(y, yPred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')'''
    # plt.show()
    return metrics.r2_score(y, yPred), yPred ,coef

