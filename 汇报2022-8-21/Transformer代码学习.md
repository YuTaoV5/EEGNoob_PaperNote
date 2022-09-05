# Transformer代码学习
## pineline里面的model
https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220821173647.png#pic_center%20=400x)
## 
bert input embedding：一种查表操作（lookup table）
查表
token embeddings：30522*768
segment embeddings：2*768
position embeddings: 512*768
后处理
layer norm
dropout

tokenizer 轻易不会将一个词处理为 [UNK] (100)
基于词汇表，tokenize, encode, decode 一体
tokenize：word => token(s)，将word尽可能地映射为 vocab 中的 keys
encode: token => id
decode: id => token => word
encode 完了之后也不是终点（word），decode 还要能很好地将 id 还原，尽可能与输入的 word 对齐；

tokenizer：服务于 model input
len(input_ids) == len(attention_mask)
tokenizer(test_senteces[0], ): tokenizer.__call__：encode
tokenizer.encode == tokenizer.tokenize + tokenizer.convert_tokens_to_ids
tokenizer.decode
tokenizer 工作的原理其实就是 tokenizer.vocab：字典，存储了 token => id 的映射关系
tokenizer.special_tokens_map
attention mask 与 padding 相匹配；
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220821223946.png#pic_center%20=400x)
outputs[0] == outputs[2][-1]
outputs[1] == model.pooler(outputs[2][-1])
outputs[2][0] == model.embeddings(token_input['input_ids'], token_input['token_type_ids'])
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220822135632.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220824094602.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220824094619.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220824094714.png#pic_center%20=400x)
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220824094751.png#pic_center%20=400x)
