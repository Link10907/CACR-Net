from torch.utils.data import Dataset


import torch

import os

import numpy as np
import open3d as o3d





def pc_normalize(pc):
    mask = np.any(pc != 0, axis=1)
    valid_points = pc[mask]
    centroid = np.mean(valid_points, axis=0)

    # 将点云平移，使有效点的质心位于原点
    pc[mask] = pc[mask] - centroid

    # 计算有效点到原点的最大距离，用于缩放
    m = np.max(np.sqrt(np.sum(pc[mask] ** 2, axis=1)))

    # 将有效点缩放到单位球内（使最大距离为 1）
    pc[mask] = pc[mask] / m

    return pc, centroid, m

    # # 过滤掉坐标为 (0, 0, 0) 的点
    # mask = np.any(pc != 0, axis=1)  # 创建一个掩码，标记不为 (0, 0, 0) 的点
    # valid_points = pc[mask]  # 仅保留有效点（不为 (0, 0, 0) 的点）
    # valid_points_set=valid_points[:,0:3]
    # valid_points_nol=valid_points[:,3:6]
    # # 计算有效点的质心
    # centroid = np.mean(valid_points_set, axis=0)
    #
    # # 将点云平移，使有效点的质心位于原点
    # pc[mask,0:3] = pc[mask,0:3] - centroid
    #
    # # 计算有效点到原点的最大距离，用于缩放
    # m = np.max(np.sqrt(np.sum(valid_points_set ** 2, axis=1)))
    #
    # # 将有效点缩放到单位球内（使最大距离为 1）
    # pc[mask,0:3] = pc[mask,0:3] / m
    #
    # return pc,centroid ,m








import trimesh
import json
def farthest_point_sample_torch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3] - single point cloud (Tensor, on GPU or CPU)
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint] (Tensor)
    """
    xyz = xyz.to(torch.float32)
    N, C = xyz.shape
    device = xyz.device  # 保持输入数据设备一致
    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), device=device).item()  # 随机初始化第一个点

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, -1)  # 当前中心点
        dist = torch.sum((xyz - centroid) ** 2, dim=1)  # 距离平方
        mask = dist < distance
        distance[mask] = dist[mask]  # 更新最近距离
        farthest = torch.argmax(distance).item()  # 找到最远的点索引

    return centroids


def index_points_np(points, idx):
    """
    Input:
        points: input points data, [N, C] - point cloud with N points, each with C features
        idx: sample index data, [S] - indices of selected points
    Return:
        new_points: indexed points data, [S, C] - selected points based on idx
    """
    new_points = points[idx]
    return new_points

def compute_normals(points):
    """
    计算点云的法向量
    :param points: 点云数据，形状为 [N, 3]
    :return: 法向量，形状为 [N, 3]
    """
    # 使用 Open3D 计算法向量
    centroid = np.mean(points, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 将点云数据传入 Open3D
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))  # 估算法线
    normals = np.asarray(pcd.normals)  # 获取法向量
    for i in range(len(points)):
        # 计算从点到质心的向量
        vec_to_centroid = centroid - points[i]

        # 计算法向量和质心方向向量的点积
        dot_product = np.dot(normals[i], vec_to_centroid)

        # 如果点积为负，说明法向量朝内，反转法向量
        if dot_product > 0:
            normals[i] = -normals[i]
    return normals
class Teeth3D(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.list_of_points = []
        self.list_of_labels=[]
        for file in os.listdir(self.data_path):
            for dir in os.listdir(os.path.join(data_path,file)):
                if dir.endswith('teeth.txt'):
                    point=np.loadtxt(os.path.join(data_path,file,dir))
                else:
                    label=np.loadtxt(os.path.join(data_path,file,dir)).astype(int)

            self.list_of_points.append(point)
            self.list_of_labels.append(label)
    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, index):
        point_set = self.list_of_points[index].copy()
        point_set, _, scale = pc_normalize(point_set)
        labels = self.list_of_labels[index]

        # 将点云重塑为 (16, 1024, 3)，每个牙齿对应一组点云
        point_set = point_set.reshape(16, 1024, 6)
        labels = labels.astype(np.int32)

        # 获取所有非零编号non_zero_labels % 10 == 3
        non_zero_labels = labels[(labels != 0)]

        # 随机选择一个非零编号
        random_label = np.random.choice(non_zero_labels)

        # 找到对应编号的索引
        random_label_index = np.where(labels == random_label)[0][0]

        # 提取选中牙齿的点云
        selected_tooth_points = point_set[random_label_index].copy()
        point_set[random_label_index] = np.zeros_like(selected_tooth_points)
        # normal = compute_normals(selected_tooth_points)
        # selected_tooth_points = np.hstack([selected_tooth_points, normal])

        point_set = point_set.reshape(16 * 1024, 6)

        return point_set, selected_tooth_points

class Teeth3D_nopc(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.list_of_points = []
        self.list_of_labels=[]
        for file in os.listdir(self.data_path):
            for dir in os.listdir(os.path.join(data_path,file)):
                if dir.endswith('teeth.txt'):
                    point=np.loadtxt(os.path.join(data_path,file,dir))
                else:
                    label=np.loadtxt(os.path.join(data_path,file,dir)).astype(int)

            self.list_of_points.append(point)
            self.list_of_labels.append(label)
    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, index):
        point_set = self.list_of_points[index].copy()
        totall_nopc=point_set.copy()
        point_set, m, scale = pc_normalize(point_set)
        labels = self.list_of_labels[index]

        # 将点云重塑为 (16, 1024, 3)，每个牙齿对应一组点云
        point_set = point_set.reshape(16, 1024, 3)
        labels = labels.astype(np.int32)

        # 获取所有非零编号non_zero_labels % 10 == 3
        non_zero_labels = labels[(labels != 0)]

        # 随机选择一个非零编号
        random_label = np.random.choice(non_zero_labels)

        # 找到对应编号的索引
        random_label_index = np.where(labels == random_label)[0][0]

        # 提取选中牙齿的点云
        selected_tooth_points = point_set[random_label_index].copy()
        point_set[random_label_index] = np.zeros_like(selected_tooth_points)
        totall_nopc = totall_nopc.reshape(16, 1024, 3)
        totall_nopc[random_label_index]=np.zeros_like(selected_tooth_points)
        # normal = compute_normals(selected_tooth_points)
        # selected_tooth_points = np.hstack([selected_tooth_points, normal])

        point_set = point_set.reshape(16 * 1024, 3)
        totall_nopc = totall_nopc.reshape(16*1024, 3)
        return point_set, selected_tooth_points,totall_nopc
if __name__ == '__main__':

    path=r'D:\2023\csl\diffusion-point-cloud-main\diffusion-point-cloud-main\data\Teeth_val_onlyteeth'
    toothdataset=Teeth3D_nopc(path)
    dataloader=torch.utils.data.DataLoader(toothdataset, batch_size=1, shuffle=False)
    for batchid,(points,label,totall) in enumerate(dataloader,0):


        point=points
        label=label
        pcd1=o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd3 = o3d.geometry.PointCloud()

        pcd1.points=o3d.utility.Vector3dVector(point.squeeze(0).numpy())
        # pcd1.normals=o3d.utility.Vector3dVector(point[:, :,3:6].squeeze(0).numpy())
        # pcd2.points = o3d.utility.Vector3dVector(tooth_points.squeeze(0).numpy())
        pcd3.points = o3d.utility.Vector3dVector(totall.squeeze(0).numpy())
        pcd2.points = o3d.utility.Vector3dVector(label.squeeze(0).numpy())
        # pcd2.normals = o3d.utility.Vector3dVector(label[:, :, 3:6].squeeze(0).numpy())
        o3d.visualization.draw_geometries([pcd3])
        o3d.visualization.draw_geometries([pcd1])
        o3d.visualization.draw_geometries([pcd2])
        # 可视化参数配置



