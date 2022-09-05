# Transformer
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813015115.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813015154.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813015323.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813015421.png#pic_center%20=400x)
### Sj是decoder hi是encoder

![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813015520.png#pic_center%20=400x)
### Query(匹配)表示用来匹配key值
### 拿qj去匹配所有K算出m个权重aj
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813015842.png#pic_center%20=400x)
### Wk和Wq Wv是权重参数,用训练数据来学习
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813020228.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813020346.png#pic_center%20=400x)

### 这里转置后K:就成了K矩阵的行向量，然后与q: j计算内积，从几何上就能得到内积越大的向量相似度越高
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220813020418.png#pic_center%20=400x)

## Attention without RNN
![Img](./FILES/transformer.md/img-20220813081145.png)
![Img](./FILES/transformer.md/img-20220813081318.png)

把xj映射到qj
一共Wk Wv Wq三个权重参数
权重a计算
![Img](./FILES/transformer.md/img-20220813081427.png)
加权平均c
![Img](./FILES/transformer.md/img-20220813081454.png)
x与c的个数一样
![Img](./FILES/transformer.md/img-20220813081731.png)
cj是列向量
![Img](./FILES/transformer.md/img-20220813081828.png)
RNN的话,会把h作为特征向量,attention是c,而c是知道整句英语的
p2抽样得到第三个德语单词x3
![Img](./FILES/transformer.md/img-20220813081915.png)
attention输出是c
![Img](./FILES/transformer.md/img-20220813082207.png)

Self-Attention without RNN
![Img](./FILES/transformer.md/img-20220813082336.png)
每个x被映射成q,k,v三个向量
![Img](./FILES/transformer.md/img-20220813082452.png)
计算m个权重a(a是m维的)
![Img](./FILES/transformer.md/img-20220813082547.png)
![Img](./FILES/transformer.md/img-20220813082609.png)
计算Context vector
![Img](./FILES/transformer.md/img-20220813082651.png)
![Img](./FILES/transformer.md/img-20220813082713.png)
这m个c向量就是Self-Attention的输出
![Img](./FILES/transformer.md/img-20220813082738.png)
改变任何一个x输出c都会改变
![Img](./FILES/transformer.md/img-20220813082856.png)
![Img](./FILES/transformer.md/img-20220813082954.png)
