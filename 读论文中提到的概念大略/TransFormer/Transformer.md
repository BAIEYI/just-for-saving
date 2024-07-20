<img src="transformer_structure.png" alt="alt text" width="200">
上面是transformer的论文结构原图，主要是两个部分，左边为编码器，右边为解码器。实际应用时，不一定是使用全部。
对于编码器：
论文中所说的N=6，当然这个可以调节。为了便利残差连接，将每个layer/sub-layer的输出维度设置为相同的（论文中为d=512）
