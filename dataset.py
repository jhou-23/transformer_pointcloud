import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import json
import torch
from pointnet_util import farthest_point_sample, pc_normalize
# torch.multiprocessing.set_start_method('spawn')

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


# class ModelNet40(Dataset):
#     def __init__(self, num_points, partition='train'):
#         self.data, self.label = load_data(partition)
#         self.num_points = num_points
#         self.partition = partition        

#     def __getitem__(self, item):
#         pointcloud = self.data[item][:self.num_points]
#         label = self.label[item]
#         if self.partition == 'train':
#             pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
#             pointcloud = translate_pointcloud(pointcloud)
#             np.random.shuffle(pointcloud)
#         return pointcloud, label

#     def __len__(self):
#         return self.data.shape[0]

class ModelNet40Old(Dataset):
    def __init__(self, root, bins, npoints=2500, split='train', data_augmentation=True, rasterize=False, cache_limit=2000):
        self.rasterize = rasterize
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.bins = bins

        fi = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s_files.json' %split), 'r')
        self.fns = list(json.load(fi))

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])
        
        self.classes = list(self.cat.keys())

        self.cache_limit = cache_limit
        self.cache = {}
        
    def __getitem__(self, index):
        if index in self.cache:
            pts, cl = self.cach[index]
        else:
            fn = self.fns[index]
            cl = self.cat[fn.split('/')[0]]
            x, y, z = self.readOFF(fn)
            pts = np.vstack([x, y, z]).T
            if len(self.cache) < self.cache_limit:
                self.cache[index] = (pts, cl)
        
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0) 
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist

        if self.data_augmentation and self.split=="train":
            point_set = random_point_dropout(point_set) 
            point_set = translate_pointcloud(point_set)
            np.random.shuffle(point_set)
        

        self.point_set = torch.from_numpy(point_set.astype(np.float32))
        cl = torch.from_numpy(np.array([cl]).astype(np.int64))
        self.point_set, discretized = self.discretize(self.bins)

        return self.point_set, discretized, cl

    def readOFF(self, fname):
        x = []
        y = []
        z = []
        with open(os.path.join(self.root, fname), 'r') as f:
            f.readline()
            stats = f.readline().strip()
            lines = int(stats.split(" ")[0])
            for i in range(lines):
                d = f.readline().strip()
                d = d.split(" ")
                x.append(float(d[0]))
                y.append(float(d[1]))
                z.append(float(d[2]))
        
        return x, y, z

    def discretize(self, bins):
        bins -= 1
        # create nx3 matrix of offsets that move points to furthest x,y,zs to 0,0,0
        offset = (torch.Tensor([torch.min(self.point_set[:,0]), torch.min(self.point_set[:,1]), torch.min(self.point_set[:,2])]).repeat(self.npoints,1))
        discretized = (self.point_set - offset)
        # scale matrix which finds largest in each direction and scales based of of bins
        scale = (float(bins) / torch.max(discretized))
        if scale.isnan() or scale.isinf():
            scale = (torch.Tensor(bins)[0])

        # discretized = discretized.mul(scale) # hadamard's product
        discretized *= scale
        discretized = torch.round(discretized) # round all elements

        if self.rasterize:
            point_list = []

            for i in range(len(self.point_set)):
                point_list.append((self.point_set[i, :], discretized[i, :]))
            # sort to raster order
            point_list = sorted(point_list, key=lambda x: [x[1][2], x[1][1], x[1][0]], reverse=False)
            discretized = []
            pc = []

            for point, discrete_point in point_list:
                pc.append(point.tolist())
                discretized.append(discrete_point.tolist())

            pc = torch.Tensor(pc).cuda()
            discretized = torch.Tensor(discretized)
        else:
            pc = self.point_set

        return pc, discretized
    
    def fps(self):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, self.npoints, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(self.npoints):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
        return centroids

    def __len__(self):
        return len(self.fns)

class ModelNet40(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

class ShapeNetDataset(Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale
        print(max(point_set))

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)



if __name__ == '__main__':
    train = ModelNet40('modelnet40')
    pc = train[1024][0]