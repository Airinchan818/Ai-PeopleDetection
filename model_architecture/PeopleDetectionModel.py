import torch 
from torch import nn 

class Patch_Embedding (nn.Module):
    def __init__(self,img_size,patch_size,embed_dim) :
      super(Patch_Embedding,self).__init__()
      self.img_size = img_size
      self.patch_size = patch_size
      self.embed_dim = embed_dim
      self.n_patch = (img_size//patch_size)**2
      self.projection_layers = nn.Conv2d(in_channels=3,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)

    def forward(self,x) :
      x = self.projection_layers(x)
      B,D,H,W = x.shape
      x = x.flatten(2)
      x = x.transpose(1,2)
      return x

class Positional_Encoding (nn.Module) :
  def __init__ (self,n_patch,embedd_dim) :
    super(Positional_Encoding,self).__init__()
    self.n_patch = n_patch
    self.embedd_dim = embedd_dim
    self.positional_encoding = nn.Parameter(torch.normal(0,0.02,size=(1,n_patch + 1,embedd_dim)))
    self.cls_token = nn.Parameter(torch.normal(0,0.02,size=(1,1,embedd_dim)))

  def forward(self,x) :
    batch = x.shape[0]
    cls_token = torch.broadcast_to(self.cls_token,(batch,1,self.embedd_dim))
    x = torch.cat((cls_token,x),dim=1)
    x = x + self.positional_encoding
    return x

class BlockTransformers (nn.Module) :
  def __init__ (self,d_Model,d_ff,n_head) :
    super(BlockTransformers,self).__init__()
    self.MHA = nn.MultiheadAttention(embed_dim=d_Model,num_heads=n_head,batch_first=True)
    self.FFN = nn.Sequential(
        nn.Linear(d_Model,d_ff),
        nn.GELU(),
        nn.Linear(d_ff,d_Model)
    )
    self.drop_out = nn.Dropout(p=0.1)
    self.drop_out2 = nn.Dropout(p=0.1)
    self.layer_norm = nn.LayerNorm(d_Model)
    self.layer_norm2 = nn.LayerNorm(d_Model)

  def forward(self,x) :
    residural = x
    x = self.layer_norm(x)
    attention,_ = self.MHA(x,x,x)
    attention = self.drop_out(attention)
    x = x + attention

    residural = x
    ffn = self.layer_norm2(x)
    ffn = self.FFN(ffn)
    ffn = self.drop_out2(ffn)
    x = residural + ffn
    return x

class MiniVisualTransformers (nn.Module) :
  def __init__(self) :
    super(MiniVisualTransformers,self).__init__()
    self.Patch_Embedding = Patch_Embedding(img_size=144,patch_size=32,embed_dim=64)
    self.Positional_Encoding = Positional_Encoding(n_patch=self.Patch_Embedding.n_patch,embedd_dim=self.Patch_Embedding.embed_dim)
    self.BT = nn.ModuleList([BlockTransformers(d_Model=64,d_ff=256,n_head=4) for _ in range(4)])

  def forward(self,x) :
    x = self.Patch_Embedding(x)
    x = self.Positional_Encoding(x)
    for block in self.BT :
      x = block(x)
    return x

class Classifier (nn.Module) :
  def __init__ (self,n_class) :
    super(Classifier,self).__init__()
    self.MiniVIT = MiniVisualTransformers()
    self.linear = nn.Linear(64,n_class)

  def forward(self,x) :
    x = self.MiniVIT(x)
    x = x[:,0,:]
    x = self.linear(x)
    return x
  
class SMVIT_M (nn.Module) :
  def __init__(self) :
    super().__init__()
    self.patch_embedding = Patch_Embedding(img_size=384,patch_size=24,embed_dim=128)
    self.posEmbedding = Positional_Encoding(n_patch=self.patch_embedding.n_patch,embedd_dim=128)
    self.block1 = BlockTransformers(d_Model=128,d_ff=256,n_head=4)
    self.block2 = BlockTransformers(d_Model=128,d_ff=256,n_head=4)
    self.block3 = BlockTransformers(d_Model=128,d_ff=256,n_head=4)
    self.block4 = BlockTransformers(d_Model=128,d_ff=256,n_head=4)

  def forward(self,x) :
    x = self.patch_embedding(x)
    x = self.posEmbedding(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    return x

class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(ObjectDetectionModel, self).__init__()
        self.vit = SMVIT_M()
        self.anchor = 1
        self.coef = 5
        self.outputs = self.anchor * (self.coef + num_classes)
        self.pred = nn.Conv2d(128, self.outputs,1)
        self.numclass = num_classes
    def forward(self,x) :
      x = self.vit(x)
      x = x[:,1:,:]
      B,N,C = x.shape
      H = W = int (N**0.5)
      x = x.permute(0,2,1)
      x = x.reshape(B,C,H,W)
      x =  self.pred(x)
      x = x.permute(0,2,3,1)
      return x.view(B,H,W,self.anchor, 5 + self.numclass)