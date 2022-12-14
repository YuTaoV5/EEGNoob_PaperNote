# MLP
一、MLP神经网络的结构和原理

理解神经网络主要包括两大内容，一是神经网络的结构，其次则是神经网络的训练和学习，其就好比我们的大脑结构是怎么构成的，而基于该组成我们又是怎样去学习和识别不同事物的，这次楼主主要讲解第一部分，而训练和学习则放到后续更新中。

神经网络其实是对生物神经元的模拟和简化，生物神经元由树突、细胞体、轴突等部分组成。树突是细胞体的输入端，其接受四周的神经冲动；轴突是细胞体的输出端，其发挥传递神经冲动给其他神经元的作用，生物神经元具有兴奋和抑制两种状态，当接受的刺激高于一定阈值时，则会进入兴奋状态并将神经冲动由轴突传出，反之则没有神经冲动。
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220814184446.png#pic_center%20=400x)


我们基于生物神经元模型可得到多层感知器MLP的基本结构，最典型的MLP包括包括三层：==输入层、隐层和输出层，MLP神经网络不同层之间是全连接的==（全连接的意思就是：上一层的任何一个神经元与下一层的所有神经元都有连接）。
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220814184510.png#pic_center%20=400x)

![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220814184521.png#pic_center%20=400x)

由此可知，神经网络主要有三个基本要素：权重、偏置和激活函数

权重：神经元之间的连接强度由权重表示，权重的大小表示可能性的大小

偏置：偏置的设置是为了正确分类样本，是模型中一个重要的参数，即保证通过输入算出的输出值不能随便激活。

激活函数：起非线性映射的作用，其可将神经元的输出幅度限制在一定范围内，一般限制在（-1~1）或（0~1）之间。最常用的激活函数是Sigmoid函数，其可将（-∞，+∞）的数映射到（0~1）的范围内。
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220814184527.png#pic_center%20=400x)


激活函数还有tanh和ReLU等函数，tanh是Sigmoid函数的变形，tanh的均值是0，在实际应用中有比Sigmoid更好的效果；ReLU是近来比较流行的激活函数，当输入信号小于0时，输出为0；当输入信号大于0时，输出等于输入；具体采用哪种激活函数需视具体情况定。
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220814184538.png#pic_center%20=400x)


从上面可知下层单个神经元的值与上层所有输入之间的关系可通过如下方式表示，其它以此类推。

![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220814184547.png#pic_center%20=400x)

MLP的最经典例子就是数字识别，即我们随便给出一张上面写有数字的图片并作为输入，由它最终给出图片上的数字到底是几。
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220814184601.png#pic_center%20=400x)

对于一张写有数字的图片，我们可将其分解为由28*28=784个像素点构成，每个像素点的值在（0~1）之间，其表示灰度值，值越大该像素点则越亮，越低则越暗，以此表达图片上的数字并将这786个像素点作为神经网络的输入。