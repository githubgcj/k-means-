import matplotlib.pyplot as plt
import numpy as np
#实现k-means聚类
fileName = "G:/硕士/研一下/DataMining/k-means聚类数据/K-means聚类数据/K-means聚类数据/Test_data.txt"
seedNum = 4


# 根据seedNum生成初始种子
def creatSeed(data):
    upper = data.max(0)
    a = np.random.rand()
    b = np.random.rand()
    seeds = []
    for i in range(seedNum):
        seed = []
        a = np.random.rand()
        b = np.random.rand()
        seed.append(a * upper[0] * 2)
        seed.append(b * upper[1] * 2)
        seeds.append(seed)
    seeds = np.array(seeds)
    return seeds


# 判断节点的簇是否发生了改变
def isChange(labels, label):
    num = sum(label != labels)
    if num == 0:
        print(0)
        return True
    else:
        print(1)
        return False


# 更新每个节点所属点的簇
def updateLabel(data, seeds):
    label = np.full((1, len(data)), -1).flatten()
    maxdis = (data.max() - data.min()) ** 2
    dis = np.full((1, len(data)), maxdis).flatten()
    for i in range(seedNum):
        tem = ((data - seeds[i]) ** 2).sum(1)
        temp = tem < dis
        for j in range(len(data)):
            if temp[j]:
                label[j] = i
                dis[j] = tem[j]
    return label


# 重新计算中心节点坐标
def updateSeeds(data, label):
    sumClass = np.zeros((seedNum, 2))
    num = np.zeros(seedNum)
    for i in range(len(data)):
        sumClass[label[i]] = sumClass[label[i]] + data[i]
        num[label[i]] = num[label[i]] + 1
    for i in range(seedNum):
        if num[i] != 0:
            sumClass[i] = sumClass[i] / num[i]
    return sumClass


def main():
    data = np.loadtxt(fileName, dtype=np.float, delimiter='\t')
    labels = np.full((1, len(data)), -1).flatten()
    seeds = creatSeed(data)
    flag = 0
    # 不同簇的颜色
    colorSeq = ['g', 'r', 'b', 'y']
    while (True):
        flag = flag + 1
        print("第 %d 轮" % (flag))
        label = updateLabel(data, seeds)
        seeds = updateSeeds(data, label)
        for i in range(len(data)):
            plt.scatter(data[i][0], data[i][1], color=colorSeq[label[i]])
        for i in range(seedNum):
            plt.scatter(seeds[i][0], seeds[i][1], color='black', marker='*')
        plt.title('第%d轮' % flag)
        plt.show()
        if isChange(labels, label):
            break
        labels = label.copy()


if __name__ == '__main__':
    main()