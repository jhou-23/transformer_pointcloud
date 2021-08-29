import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import copy
import math
from torch.autograd import Variable
from pointnet_util import farthest_point_sample, index_points, square_distance
import provider

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value):
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, dropout):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadedAttention(d_model=d_model, h=n_head, dropout=drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        x_res = x
        x = self.attention(query=x, key=x, value=x)
        
        x = self.norm1(x + x_res)
        x = self.dropout1(x)

        x_res = x
        x = self.ffn(x)
        
        x = self.norm2(x + x_res)
        x = self.dropout2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, n_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Classifier(nn.Module):
    def __init__(self, d_model, out_size, drop_prob):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(d_model, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=drop_prob)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=drop_prob)
        self.linear3 = nn.Linear(256, out_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class PointTransformer(nn.Module):
    def __init__(self, config, num_classes=40):
        super(PointTransformer, self).__init__()
        config = config["model"]["components"]
        self.config = config
        ffn_hidden = config["ffn_hidden"]
        n_head = config["num_heads"]
        drop_prob = config["dropout_p"]
        n_layers = config["num_layers"]
        d_model = 3
    
        if config["input_embedding"]["included"]:
            d_model = config["input_embedding"]["d_model"]
            self.fc1 = nn.Conv1d(3, d_model, kernel_size=1, bias=False)
            self.fc2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm1d(d_model)
            self.bn2 = nn.BatchNorm1d(d_model)
            self.relu = nn.ReLU()

        self.encoder = Encoder(d_model, ffn_hidden, n_head, drop_prob, n_layers)
        self.classifier = Classifier(d_model, num_classes, drop_prob)
    
    def forward(self, x):
        if self.config["input_embedding"]["included"]:
            x = x.permute(0, 2, 1)
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.relu(self.bn2(self.fc2(x)))
            x = x.permute(0, 2, 1)
        x = self.encoder(x)
        
        x = torch.max(x, dim=1)[0] #TODO:check max pool for both pct and this
        x = self.classifier(x)
        return x

class SinusoidalPositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d):
        super(SinusoidalPositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        self.d = d
        self.out_size = d * 3
        
    def forward(self, xyz):
        # xyz = 2500 * (xyz + 1)
        x = xyz[:,:,0].unsqueeze(2).cuda()
        y = xyz[:,:,1].unsqueeze(2).cuda()
        z = xyz[:,:,2].unsqueeze(2).cuda()
        pex = torch.zeros(x.shape[0],x.shape[1], self.d).cuda()
        pey = torch.zeros(y.shape[0],y.shape[1], self.d).cuda()
        pez = torch.zeros(z.shape[0],z.shape[1], self.d).cuda()
        pos_x = x.cuda()
        pos_y = y.cuda()
        pos_z = z.cuda()
        div_term = torch.exp(torch.arange(0, self.d, 2) *
                             -(math.log(10000.0) / self.d)).cuda()
        batch_div_term = div_term.repeat(x.shape[0], 1, 1).cuda()
        pex[:,:, 0::2] = torch.sin(pos_x.bmm(batch_div_term)).cuda()
        pex[:,:, 1::2] = torch.cos(pos_x.bmm(batch_div_term)).cuda()

        pey[:,:, 0::2] = torch.sin(pos_y.bmm(batch_div_term)).cuda()
        pey[:,:, 1::2] = torch.cos(pos_y.bmm(batch_div_term)).cuda()

        pez[:,:, 0::2] = torch.sin(pos_z.bmm(batch_div_term)).cuda()
        pez[:,:, 1::2] = torch.cos(pos_z.bmm(batch_div_term)).cuda()

        pe = torch.cat([pex, pey, pez], dim=-1).cuda()
        return pe


class SinusoidalPositionalEncodingPadded(nn.Module):
    "Implement the PE function."
    def __init__(self, d, d_model):
        super(SinusoidalPositionalEncodingPadded, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        self.d = d
        self.out_size = d * 3
        self.difference = d_model - self.out_size
        if self.difference < 0:
            print("Error: PE not possible because d-model:%d is smaller than PE-size:%d" % (d_model, self.out_size))
            exit()
        
    def forward(self, xyz):
        xyz = 2500 * (xyz + 1)
        x = xyz[:,:,0].unsqueeze(2).cuda()
        y = xyz[:,:,1].unsqueeze(2).cuda()
        z = xyz[:,:,2].unsqueeze(2).cuda()
        pex = torch.zeros(x.shape[0],x.shape[1], self.d).cuda()
        pey = torch.zeros(y.shape[0],y.shape[1], self.d).cuda()
        pez = torch.zeros(z.shape[0],z.shape[1], self.d).cuda()
        pos_x = x.cuda()
        pos_y = y.cuda()
        pos_z = z.cuda()
        div_term = torch.exp(torch.arange(0, self.d, 2) *
                             -(math.log(10000.0) / self.d)).cuda()
        batch_div_term = div_term.repeat(x.shape[0], 1, 1).cuda()
        pex[:,:, 0::2] = torch.sin(pos_x.bmm(batch_div_term)).cuda()
        pex[:,:, 1::2] = torch.cos(pos_x.bmm(batch_div_term)).cuda()

        pey[:,:, 0::2] = torch.sin(pos_y.bmm(batch_div_term)).cuda()
        pey[:,:, 1::2] = torch.cos(pos_y.bmm(batch_div_term)).cuda()

        pez[:,:, 0::2] = torch.sin(pos_z.bmm(batch_div_term)).cuda()
        pez[:,:, 1::2] = torch.cos(pos_z.bmm(batch_div_term)).cuda()

        pe = torch.cat([pex, pey, pez], dim=-1).cuda()
        padding = torch.zeros(pe.shape[0], pe.shape[1], self.difference).cuda()
        pe = torch.cat([pe, padding], dim=-1).cuda()
        return pe


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, bins, d_q):
        super(AbsolutePositionalEmbedding, self).__init__()
        dir_length = d_q // 3
        self.pad_length = d_q - (dir_length * 3)
        self.emb_x = nn.Parameter(torch.Tensor(bins, dir_length))
        self.emb_y = nn.Parameter(torch.Tensor(bins, dir_length))
        self.emb_z = nn.Parameter(torch.Tensor(bins, dir_length))
        nn.init.xavier_uniform_(self.emb_x)
        nn.init.xavier_uniform_(self.emb_y)
        nn.init.xavier_uniform_(self.emb_z)
    
    def forward(self, discrete):
        # discrete: batch x npoint x 3
        x_idx = discrete[:,:,0].type(torch.LongTensor).cuda()
        y_idx = discrete[:,:,1].type(torch.LongTensor).cuda()
        z_idx = discrete[:,:,2].type(torch.LongTensor).cuda()

        x_pe = self.emb_x[x_idx]
        y_pe = self.emb_y[y_idx]
        z_pe = self.emb_z[z_idx]

        zero_pad = torch.zeros(discrete.shape[0], discrete.shape[1], self.pad_length).cuda()

        lpe = torch.cat([x_pe, y_pe, z_pe, zero_pad], dim=-1)
        return lpe


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, bins, d_q):
        super(RelativePositionalEmbedding, self).__init__()
        self.bins = bins
        self.x_lt = nn.Parameter(torch.Tensor(2 * self.bins - 1, d_q))
        self.y_lt = nn.Parameter(torch.Tensor(2 * self.bins - 1, d_q))
        self.z_lt = nn.Parameter(torch.Tensor(2 * self.bins - 1, d_q))
        nn.init.xavier_uniform_(self.x_lt)
        nn.init.xavier_uniform_(self.y_lt)
        nn.init.xavier_uniform_(self.z_lt)
    
    def forward(self, q, discrete):
        x_discrete = discrete[:,:,0].type(torch.LongTensor).squeeze().cuda()
        y_discrete = discrete[:,:,1].type(torch.LongTensor).squeeze().cuda()
        z_discrete = discrete[:,:,2].type(torch.LongTensor).squeeze().cuda()
        
        x_idx = ((x_discrete[:, None, :] - x_discrete[:, :,None]).type(torch.LongTensor).cuda() + self.bins - 1).cuda() 
        y_idx = ((y_discrete[:, None, :] - y_discrete[:, :,None]).type(torch.LongTensor).cuda() + self.bins - 1).cuda()
        z_idx = ((z_discrete[:, None, :] - z_discrete[:, :,None]).type(torch.LongTensor).cuda() + self.bins - 1).cuda()

        qx_lt = q.bmm(self.x_lt.unsqueeze(0).repeat(q.shape[0], 1, 1).permute(0,2,1)).cuda()
        qy_lt = q.bmm(self.y_lt.unsqueeze(0).repeat(q.shape[0], 1, 1).permute(0,2,1)).cuda()
        qz_lt = q.bmm(self.z_lt.unsqueeze(0).repeat(q.shape[0], 1, 1).permute(0,2,1)).cuda()

        qx_r = torch.gather(qx_lt, 2, x_idx).cuda()
        qy_r = torch.gather(qy_lt, 2, y_idx).cuda()
        qz_r = torch.gather(qz_lt, 2, z_idx).cuda()

        qr = qx_r + qy_r + qz_r
        return qr


class StaticGaussianRelativePositionEmbedding(nn.Module):
    def __init__(self, sigma, npoints):
        super(StaticGaussianRelativePositionEmbedding, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        dc = -1 / (2 * (self.sigma**2))
        cc = 1 / ((2 * math.pi * self.sigma) ** 0.5)
        dists = self.pairwise_distances(x).cuda()
        pe = cc * torch.exp(dc * dists)
        return pe.cuda()
    
    def pairwise_distances(self, x):
        '''
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(2).unsqueeze(2)
        y_t = torch.transpose(x, 1, 2)
        y_norm = x_norm.permute(0,2,1)
        
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf) ** 0.5


class GaussianRelativePositionEmbedding(nn.Module):
    def __init__(self, npoints):
        super(GaussianRelativePositionEmbedding, self).__init__()
        self.sigma = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        dc = -1 / (2 * (self.sigma**2)).cuda()
        cc = 1 / ((2 * math.pi * self.sigma) ** 0.5).cuda()
        dists = self.pairwise_distances(x).cuda()
        pe = cc * torch.exp(dc * dists)
        return pe
    
    def pairwise_distances(self, x):
        '''
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(2).unsqueeze(2)
        y_t = torch.transpose(x, 1, 2)
        y_norm = x_norm.permute(0,2,1)
        
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf) ** 0.5

class SA_Layer(nn.Module):
    def __init__(self, config, channels):
        super(SA_Layer, self).__init__()
        self.config = config
        self.encode_position = self.config["model"]["components"]["input_embedding"]["pos_enc"]
        self.att_config = config["model"]["components"]["attention"]
        if self.encode_position == "ape":
            self.pos_encoder = AbsolutePositionalEmbedding(config["dataset"]["bins"], self.att_config["d_qk"])
        elif self.encode_position == "rpe":
            self.pos_encoder = RelativePositionalEmbedding(config["dataset"]["bins"], self.att_config["d_qk"])
        elif self.encode_position == "grpe":
            self.pos_encoder = GaussianRelativePositionEmbedding(self.config["dataset"]["num_points"])
        elif self.encode_position == "sgrpe":
            self.pos_encoder = StaticGaussianRelativePositionEmbedding(self.config["model"]["components"]["input_embedding"]["sigma"], self.config["dataset"]["num_points"])

        self.q_conv = nn.Conv1d(channels, self.att_config["d_qk"], 1, bias=False)
        self.k_conv = nn.Conv1d(channels, self.att_config["d_qk"], 1, bias=False)
        if self.att_config["shared_weights"]:
            self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, self.att_config["d_v"], 1)
        self.trans_conv = nn.Conv1d(self.att_config["d_v"], channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, discrete, xyz):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k) # b, n, n 
        if self.encode_position == "ape":
            pe = self.pos_encoder(discrete)
            energy = energy + x_q.bmm(pe.permute(0, 2, 1))
        elif self.encode_position == "rpe":
            qr = self.pos_encoder(x_q, discrete)
            energy = energy + qr

        if self.att_config["scale"] == "root":
            energy = energy / (energy.size(-1) ** 0.5)
        attention = self.softmax(energy)
        if self.att_config["scale"] == "l1":
            attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        if self.encode_position == "grpe":
            attention = attention + self.pos_encoder(xyz)
        elif self.encode_position == "sgrpe":
            attention = attention + self.pos_encoder(xyz)

        x_r = torch.bmm(x_v, attention) # b, c, n 

        if self.att_config["offset"]:
            x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        else:
            x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        x = x + x_r
        return x


class PCT(nn.Module):
    def __init__(self, config, output_channels=40):
        super(PCT, self).__init__()
        self.config = config
        self.raw = 3
        self.in_channel = 3
        if self.config["model"]["components"]["input_embedding"]["included"]:
            # if self.config["model"]["components"]["input_embedding"]["pos_enc"] == "sinusoidal":
            #     self.position_enc = SinusoidalPositionalEncoding(32)
            #     self.raw = self.position_enc.out_size
            if self.config["model"]["components"]["input_embedding"]["neighbor"]:
                self.in_channel = 256
                self.conv1 = nn.Conv1d(self.raw, 64, kernel_size=1, bias=False)
                self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm1d(64)
                self.bn2 = nn.BatchNorm1d(64)
                self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
                self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
                self.conv3 = nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1, bias=False)
                self.conv4 = nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm1d(self.in_channel)
                self.bn4 = nn.BatchNorm1d(self.in_channel)
            else:
                self.in_channel = self.config["model"]["components"]["input_embedding"]["d_model"]
                self.conv1 = nn.Conv1d(self.raw, self.in_channel, kernel_size=1, bias=False)
                self.conv2 = nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm1d(self.in_channel)
                self.bn2 = nn.BatchNorm1d(self.in_channel)

            if self.config["model"]["components"]["input_embedding"]["pos_enc"] == "sinusoidal":
                D = self.in_channel // 3
                self.position_enc = SinusoidalPositionalEncodingPadded(D, self.in_channel)

        self.sa = nn.ModuleList(SA_Layer(self.config, self.in_channel) for l in range(self.config["model"]["components"]["num_layers"]))

        if self.config["model"]["components"]["concat"]:
            self.conv_fuse = nn.Sequential(nn.Conv1d((self.config["model"]["components"]["num_layers"] + self.config["model"]["components"]["input_embedding"]["neighbor"]) * self.in_channel, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

            self.linear1 = nn.Linear(1024, 512, bias=False)
        else:
            self.linear1 = nn.Linear(self.in_channel, 512, bias=False)

        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.config["model"]["components"]["dropout_p"])
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.config["model"]["components"]["dropout_p"])
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()
        
    def forward(self, x, discrete):
        xyz = x
        if self.config["model"]["components"]["input_embedding"]["pos_enc"] == "sinusoidal":
            pe = self.position_enc(x)
        
        x = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        if self.config["model"]["components"]["input_embedding"]["included"]:
            x = self.relu(self.bn1(self.conv1(x))) # B, D, N
            x = self.relu(self.bn2(self.conv2(x)))

            if self.config["model"]["components"]["input_embedding"]["neighbor"]:
                x = x.permute(0, 2, 1)
                new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)         
                feature_0 = self.gather_local_0(new_feature)
                feature = feature_0.permute(0, 2, 1)
                new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) 
                feature_1 = self.gather_local_1(new_feature)

            if self.config["model"]["components"]["input_embedding"]["pos_enc"] == "sinusoidal":
                x += pe.permute(0,2,1)

        if self.config["model"]["components"]["input_embedding"]["neighbor"]:
            x_temp = self.relu(self.bn3(self.conv3(feature_1)))
            x_temp = self.relu(self.bn4(self.conv4(x_temp)))
            if self.config["model"]["components"]["input_embedding"]["neighbor"]: #TODO: Make sure it makes sense
                _, discrete = provider.discretize(new_xyz.cuda(), self.config["dataset"]["bins"])
        else:
            x_temp = x
        
        xs = []
        for layer in self.sa:
            x_temp = layer(x_temp, discrete, xyz)
            xs.append(x_temp)
        
        if self.config["model"]["components"]["concat"]:
            if self.config["model"]["components"]["input_embedding"]["neighbor"]:
                xs.append(feature_1)
            x = torch.cat(xs, dim=1)
            x = self.conv_fuse(x)
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        if self.config["model"]["components"]["pool"] == "max":
            x = torch.max(x, 2)[0]
        else:
            x = x.mean(dim = 2)
            
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
