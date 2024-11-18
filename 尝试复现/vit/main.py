import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):#将输入转换为元组形式，如果输入本身是元组，则直接返回，如果不是，则将其与自身组成一个元组。
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)#一个标志，如果头数为1且每个头的维度等于输入维度，则不需要最后的线性层。

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)#nn.Dropout()接受的参数为丢弃率

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()#一个恒等映射（identity mapping）层，即输出与输入完全相同

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        #假设 self.to_qkv(x) 的输出张量的形状是 [batch_size, sequence_length, 3 * embedding_dim]，
        #chunk(3, dim = -1) 将沿着最后一个维度（embedding_dim）
        #将这个张量分割成三个形状为 [batch_size, sequence_length, embedding_dim] 的张量。
        #这三个张量分别对应于查询（Q）、键（K）和值（V）。此时qkv是一个包含三个张量的元组。
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        #map函数接收两个参数，第一个是一个函数，第二个是一个可迭代对象（在这里是qkv）。map会应用这个函数到qkv中的每一个元素上，并返回一个迭代器。
        #map中的第一项是一个匿名函数（lambda函数），它接收一个参数t，并返回rearrange函数的调用结果。rearrange函数通常用于重新排列张量的维度。
        #对于rearrange函数中，t是输入的张量。'b n (h d) -> b h n d'是一个维度重排的公式。它指定了如何从输入张量的维度转换到输出张量的维度。
        #举例：如果操作前维度为[10, 197, 512]，也就是[10, 197, 8*64]，操作后就可以变成[10, 8, 197, 64]
        #h = self.heads：这是在rearrange函数调用时传递的参数，传递了多头注意力中的头数。
        #b是batch批量数，n是序列长度（vit的patch数），h是多头注意力的头数，d是每个头的特征维度。
        #经过这一句后，q、k、v这三个原来是在qkv中的张量进行了命名以及，分别对三者进行了维度重排。

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #q、k之间进行矩阵乘法。k.transpose(-1, -2)是将键张量k的最后两个维度进行转置。这是因为在计算点积时，需要确保q和k的最后一个维度是兼容的（即它们的维度大小必须相等）

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')#把这个维度调回来，接上面的例子就是再次变成[10, 197, 512]
        return self.to_out(out)
        #可以看出来，这里多头的实现并不是和论文中提到的完全一样，拆开再拼接什么的，而是通过维度的调整而在数学上达到一样的效果（就是让人看起来头大点~~~）

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
    #在 __init__ 方法的参数列表中，* 表示所有随后的参数必须以关键字参数的形式传递，即通过参数名显式指定值。
    #例如，用的时候只能以vit_model = ViT(image_size=224,…………）这种形式，而不能vit_model = ViT(224…………)这种形式。
    #使用 * 的好处是它提高了代码的可读性，使得在调用函数时更清楚每个参数的含义，同时避免了因参数顺序导致的错误
        super().__init__()
        image_height, image_width = pair(image_size)#好聪明的方法，这样就可以做好设置了。
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        #使用全局池化或者添加额外cls token。

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()#nn.Identity() 是一个简单的模块，它不做任何改变地返回其输入。这个模块通常用于模型定义中，当你需要一个模块，但又不希望对输入执行任何操作时。

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)#repeat 函数在 PyTorch 中用于创建一个新的张量，该张量是通过重复原始张量的某些维度来形成的。这里重复了b次。
        x = torch.cat((cls_tokens, x), dim=1)#将cls与x进行拼接。
        x += self.pos_embedding[:, :(n + 1)]#虽然是加上去的，但是self.pos_embedding仍然可以进行训练。
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]#如果不采用全局平均池化，就用cls，这里写的十分简洁，就是找了x的序列中的第一个token，也就是cls token作为输出。

        x = self.to_latent(x)
        return self.mlp_head(x)