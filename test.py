# 域名检测
# train.txt, test.txt, result.txt
# 每条域名选取三种特征：域名长度，域名中的数字数量，域名字符熵
# 机器学习采用随机森林算法

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math
from pandas.core.frame import DataFrame


# 计算字符熵
def LettersEntropy(str):
    # 统计每个字符出现的次数
    letters, counts = np.unique(list(str), return_counts=True)
    total = sum(counts)
    entropy = 0.0
    for i in range(len(counts)):
        prob = counts[i]/total
        entropy += - prob * math.log(2, prob)
    return entropy

def NumCollect(str):
    # 统计每条域名中的数字个数
    total = 0
    for i in str:
        if i.isdigit():
            total +=1
    return total

if __name__ == '__main__':
    # 创建3个特征列表，1个种类列表
    DomainLen = []
    Numbers = []
    Entropy = []
    Type = []
    # 读取train.txt文件
    f = open(r'train.txt')
    for line in f:
        line = line.strip()
        if line == "":
            continue
        tokens = line.split(",")
        domain = tokens[0]
        label = tokens[1]
        DomainLen.append(len(domain))
        Numbers.append(NumCollect(domain))
        Entropy.append(LettersEntropy(domain))
        if label == 'notdga':
            Type.append(0)
        else:
            Type.append(1)
    traindata = {'Length': DomainLen, 'NumInDomain': Numbers, 'Entropy': Entropy, 'Type': Type}
    traindata = DataFrame(traindata)
    # 设置训练数据集
    y = traindata.Type
    x = traindata.drop('Type', axis=1)
    xtrain = x
    ytrain = y
    # 读取test.txt文件
    testDomainLen = []
    testNumbers = []
    testEntropy = []
    testType = []
    testDomainName = []
    TestFile = open(r'test.txt')
    for line in TestFile:
        line = line.strip()
        if line == "":
            continue
        testDomainName.append(line)
        testDomainLen.append(len(line))
        testNumbers.append(NumCollect(line))
        testEntropy.append(LettersEntropy(line))
    testdata = {'Length': testDomainLen, 'NumInDomain': testNumbers, 'Entropy': testEntropy}
    testdata = DataFrame(testdata)
    # 设置测试数据集
    xtest = testdata
    # 实例化
    rfc = RandomForestClassifier()
    # 用训练集数据训练模型
    rfc = rfc.fit(xtrain, ytrain)
    ytest = rfc.predict(xtest)
    for i in range(len(ytest)):
        if ytest[i] == 0:
            testType.append('notdga')
        else:
            testType.append('dga')
    with open('result.txt', 'w+') as ResultFile:
        for i in range(len(testDomainName)):
            line = testDomainName[i] + ',' + testType[i] + '\n'
            ResultFile.write(line)