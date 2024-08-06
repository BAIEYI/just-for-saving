<img src="transformer_structure.png" alt="alt text" width="300">
上面是transformer的论文结构原图，主要是两个部分，左边为编码器，右边为解码器。实际应用时，不一定是使用全部。<br>
对于编码器:<br>
论文中所说的N=6，当然这个可以调节。为了便利残差连接，将每个layer/sub-layer的输出维度设置为相同的（论文中为d=512）<br>
<img src="layer_norm.png" alt="layer_norm" width="300">
这里讲了一个layer_norm的东西（没错，在结构图中的“Add & Norm”的Norm指的就是layer_norm），以及与平时经常使用的batch_norm的区别。见图的左下即为普通的batch_norm，是对于batch中的每个特征向量进行normalization（均值变成0，方差变1）。<br>
右下即为layer_norm的理解，但结合上面的图比较好理解，是对于每个 样本 来进行normlization。<br>
<br><br>


这里说了一些注意力机制的东西这里要进行额外的基础知识补全学习了。关于注意力机制：<br>
注意力函数可以描述为将一个查询（query）和一组键值对（key-value pairs）映射到一个输出的机制。其中，查询、键、值和输出在这个过程中都是向量形式。<br>
而注意力函数的输出为一个针对value的加权和（所以说输出的维度和value的维度应该是相当的）。
Q（query）、K（key）、V（value）