# RNN
Stacked RNN（多层RNN）
embeding层参数过多造成overfiting
Bidirectional RNN（双向RNN）
Pretrain（预训练）
Tokenization(把一句话分割)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813010008.png#pic_center%20=400x)
不同level(word,char),有不同字典vocabulary(分词方法),做One-Hot-Encoding
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813010231.png#pic_center%20=400x)
Encoder的最后一个状态(是这句话的概要)给Decoder作为初始状态,得知原文数据
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813010515.png#pic_center%20=400x)
计算y与p损失函数,反向传播梯度下降修正模型
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813010818.png#pic_center%20=400x)
步骤一
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813010904.png#pic_center%20=400x)
步骤二
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813010927.png#pic_center%20=400x)

![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813011208.png#pic_center%20=400x)
Bi-LSTM encoder两个方向,decoder只需要一个方向
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813011308.png#pic_center%20=400x)

![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813011749.png#pic_center%20=400x)
多task Decoder能够提高准确度
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813011848.png#pic_center%20=400x)

Attention
避免遗忘,每次decoder更新都会看一眼encoder所有状态,计算量会增大
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813012402.png#pic_center%20=400x)
计算s0和hi的相关性分别为a1,a2....am
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813012341.png#pic_center%20=400x)
a1~am>0,且相加为一
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813012554.png#pic_center%20=400x)
计算加权平均c0对应于s0,c0=a1*h1+a2*h2......,decoder更新s1,然后重新计算新的与S1的相关性分别为a1,a2....am,计算加权平均c1对应于s1,decoder更新s2
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813013345.png#pic_center%20=400x)

![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813012732.png#pic_center%20=400x)
计算代价大
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813013940.png#pic_center%20=400x)
线代表a权重
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813014122.png#pic_center%20=400x)

![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813014828.png#pic_center%20=400x)

![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813014646.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813014653.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813014716.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813014725.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813014856.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813014959.png#pic_center%20=400x)
