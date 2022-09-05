# EEG信号分类EEGNet+Transformer



## 实验数据
每个参与者每12个单词进行25次随机实验，每种情况下总共进行300次试验。
共有13个分类输出，包括12个单词
- ambulance
- clock
- hello
- help me
- light
- pain
- stop
- thank you
- toilet
- TV
- water
- yes
- resting state
## 预处理
### 滤波+去基线
以250Hz对EEG信号进行下采样，每个试验数据进行2秒时间长度的切片。在30–120Hz的high-gamma波段，使用==五阶巴特沃斯滤波器==对EEG信号进行预处理，并通过减去每次试验开始前500 ms的平均值来==校正基线==。
### 去伪影
为了消除口腔周围肌肉活动中的EOG和EMG伪影，我们使用独立分量分析和EOG和肌电图的参考进行伪影消除方法。
## 网络结构
### CNN提取时间谱空间信息
由卷积层和可分离卷积层组成，用于提取时间谱空间信息,分类输出设置为13类。第一层的核大小与数据的采样频率相关，用于执行模拟带通滤波器的时间卷积。训练的损失函数squared hinge loss,五折 交叉验证,训练1000轮epoch
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220823230236.png#pic_center%20=400x)
将图像==分割==成固定大小的块，线性嵌入每个块，添加==position embeddings==，并将生成的矢量序列馈送到标准==Transformer encoder==。为了执行分类，我们使用标准方法向序列中添加额外的可学习分类标记==class token==。
在之后添加dropout连接LN层接上Transformer
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220823230857.png#pic_center%20=400x)
