# -*- coding: utf-8 -*-
"""
@author: Anton Wang
"""

# 3D model class

import numpy as np
import open3d as o3d
from knn import kneighbors

class a_3d_model:
    def __init__(self, filepath, saliency, lam=10.0):
        self.model_filepath=filepath
        self.load_obj_file()

        if not type(saliency) == np.ndarray:
            raise Exception('saliency has to be an ndarray.')
        if not saliency.size == self.number_of_points:
            raise Exception('saliency has to be of shape N.')
        if not saliency.dtype == np.double:
            raise Exception('saliency has to be of type double')
        if not saliency.max()==1.0: # normalize
            saliency /= saliency.max()
        self.saliency=saliency
        self.lam= lam
        self.normals = None

        self.init_KDTree()
        self.estimate_normals()
        self.calculate_plane_equations()
        self.calculate_Q_matrices()
        
    def load_obj_file(self):
        with open(self.model_filepath) as file:
            self.points = []
            self.faces = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    self.points.append((float(strs[1]), float(strs[2]), float(strs[3])))
                if strs[0] == "f":
                    self.faces.append((int(strs[1]), int(strs[2]), int(strs[3])))
        self.points=np.array(self.points)
        self.faces=np.array(self.faces)
        self.number_of_points=self.points.shape[0]
        self.number_of_faces=self.faces.shape[0]
        edge_1=self.faces[:,0:2]
        edge_2=self.faces[:,1:]
        edge_3=np.concatenate([self.faces[:,:1], self.faces[:,-1:]], axis=1)
        self.edges=np.concatenate([edge_1, edge_2, edge_3], axis=0)
        unique_edges_trans, unique_edges_locs=np.unique(self.edges[:,0]*(10**10)+self.edges[:,1], return_index=True)
        self.edges=self.edges[unique_edges_locs,:]
    
    def init_KDTree(self):
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(self.points)
        self.pc_tree = o3d.geometry.KDTreeFlann(pointcloud)

    def estimate_normals(self, max_k=16):
        '''
        Estimate normals considering a knn-neighborhood
        '''
        # TODO estimate normals from tangent plane fitted at vertex
        # self.normals = np.asarray(point_cloud.normals)
        assert self.normals.shape==(self.points.shape)

    def calculate_plane_equations(self):
        # Tangent plane estimation for point sampled surfaces
        # TODO use KDTree for faster KNN queries
        knn_points = kneighbors(self.points, k=5)
        self.edge_tangents = []
        for P in knn_points:
            p, neighbors = P[0], P[1:]  # split into relevant point and neighbors
            planes = []
            for pj in neighbors:
                ej = self.points[p] - self.points[pj]
                bj = np.cross(ej, self.normals[p])
                # "tangent" plane on this edje is spanned by ej and bj
                tj = np.cross(ej, bj)
                planes.append(tj)
            self.edge_tangents.append(planes)
        self.edge_tangents = np.array(self.edge_tangents)
        print(self.edge_tangents.shape)

        #### OLD Face based plane computation ####
        # self.plane_equ_para = []
        # for i in range(0, self.number_of_faces):
        #     # solving equation ax+by+cz+d=0, a^2+b^2+c^2=1
        #     # set d=-1, give three points (x1, y1 ,z1), (x2, y2, z2), (x3, y3, z3)
        #     point_1=self.points[self.faces[i,0]-1, :]
        #     point_2=self.points[self.faces[i,1]-1, :]
        #     point_3=self.points[self.faces[i,2]-1, :]
        #     point_mat=np.array([point_1, point_2, point_3])
        #     abc=np.matmul(np.linalg.inv(point_mat), np.array([[1],[1],[1]]))
        #     self.plane_equ_para.append(np.concatenate([abc.T, np.array(-1).reshape(1, 1)], axis=1)/(np.sum(abc**2)**0.5))
        # self.plane_equ_para=np.array(self.plane_equ_para)
        # self.plane_equ_para=self.plane_equ_para.reshape(self.plane_equ_para.shape[0], self.plane_equ_para.shape[2])
    
    def calculate_Q_matrices(self):
        # Apply saliency weight here to each Q matrix
        alpha = np.percentile(self.saliency, 30)
        print(f"The 30th percentile saliency is {alpha}")
        weight = np.where(self.saliency > alpha, self.lam*self.saliency, self.saliency)

        self.Q_matrices = []
        for i in range(0, self.number_of_points):
            point_index=i+1
            # each point is the solution of the intersection of a set of planes
            # find the planes for point_index
            face_set_index=np.where(self.faces==point_index)[0]
            Q_temp=np.zeros((4,4))
            for j in face_set_index:
                p=self.plane_equ_para[j,:]
                p=p.reshape(1, len(p))
                Q_temp=Q_temp+np.matmul(p.T, p)

            Q_temp = Q_temp*weight[i]
            self.Q_matrices.append(Q_temp)