#TODO: restructure code so that stuff is separated (follow tutorial) see line 36 and consider the modulelist
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import copy

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class AttentionHead(nn.Module):
    def __init__(self, d_in, d_qk, d_v):
        super(AttentionHead, self).__init__()
        self.wq = nn.Linear(d_in, d_qk)
        self.wk = nn.Linear(d_in, d_qk)
        self.wv = nn.Linear(d_in, d_v)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        product = q.bmm(k.transpose(1,2))
        scale = q.size(-1) ** 0.5
        scaled_softmax = self.softmax(product / scale)
        return scaled_softmax.bmm(v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_qk, d_v, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.mha = clone(AttentionHead(d_in, d_qk, d_v), num_heads)
        self.wo = nn.Linear(num_heads * d_v, d_in)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.mha], dim=-1)
        x = self.wo(x)
        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        # energy = energy / (energy.size(-1) ** 0.5)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        x = x + x_r
        return x


class ResNorm(nn.Module):
    def __init__(self, module, dimension, dropout=0.0):
        super(ResNorm, self).__init__()
        self.module = module
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        output = self.dropout(self.module(x))
        residual = output + x
        normalized = self.norm(residual)
        return normalized

class FFN(nn.Module):
    def __init__(self, d_in, d_ffn):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(d_in, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_in)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x) # watch if change
        return x

class PointCloudLayer(nn.Module):
    def __init__(self, d_in, d_ffn, d_qk, d_v, dropout, num_heads):
        super(PointCloudLayer, self).__init__()
        self.attention = ResNorm(MultiHeadAttention(d_in, d_qk, d_v, num_heads), d_in, dropout)
        self.feed_forward = ResNorm(FFN(d_in, d_ffn), d_in, dropout)

    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x
        

class PointTransformer(nn.Module):
    def __init__(self, num_layers=6, ffn_dim=2048, num_heads=12, num_classes=40, in_size=3, dropout=0.0, d_qk=64,d_v=64):
        super(PointTransformer, self).__init__()
        self.encoder = nn.ModuleList([PointCloudLayer(in_size, ffn_dim, d_qk, d_v, dropout, num_heads) for l in range(num_layers)])
        self.fc1 = nn.Sequential(nn.Linear(in_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.0))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.0))
        self.fc3 = nn.Linear(128, num_classes)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = x.mean(dim = 1)
        x = self.fc3(self.fc2(self.fc1(x)))
        return x

class PointTransformer2(nn.Module):
    def __init__(self, num_layers=6, ffn_dim=2048, num_heads=12, num_classes=40, in_size=3, extractor=128, dropout=0.0, d_qk=64, d_v=64):
        super(PointTransformer2, self).__init__()
        
        self.feat1 = nn.Linear(3, 128)
        self.bn1 = nn.LayerNorm(128)
        self.feat2 = nn.Linear(128, 128)
        self.bn2 = nn.LayerNorm(128)

        self.encoder = nn.ModuleList([PointCloudLayer(extractor, ffn_dim, d_qk, d_v, dropout, num_heads) for l in range(num_layers)])
        
        self.fc1 = nn.Sequential(nn.Linear(extractor, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.0))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.0))
        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat1(x)
        x = self.relu(self.bn1(x))
        x = self.relu(self.bn2(self.feat2(x)))
        for layer in self.encoder:
            x = layer(x)
        x = x.mean(dim = 1)
        x = self.fc1(x)
        x = self.fc3(self.fc2(x))
        return x

class PCT(nn.Module):
    def __init__(self, output_channels=40):
        super(PCT, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv_fuse(x)
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

if __name__ == '__main__':
    model = PointTransformer().cuda()
    data = torch.rand(8, 2048, 3).cuda()
    print(data)
    print(data.shape)
    out = model(data)
    print(out.shape)
