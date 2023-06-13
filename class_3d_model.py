# -*- coding: utf-8 -*-
"""
@author: Anton Wang
"""

# 3D model class

import numpy as np
import open3d as o3d
from knn import kneighbors_all

class a_3d_model:
    def __init__(self, filepath, saliency, lam=10.0):
        self.input_filepath=filepath
        self.load_pc_from_obj()

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

        # use o3d PointCloud structure
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        self.pc = pc

        # self.init_KDTree()
        self.estimate_normals()
        self.Q_matrices = [None for i in range(self.number_of_points)]
        self.calculate_Qs()
        self.apply_saliency_weight()
        # self.calculate_Q_matrices()
        
    def load_npy_file(self):
        self.points = np.load(self.input_filepath)
        self.number_of_points = len(self.points)

    def load_pc_from_obj(self):
        with open(self.input_filepath) as file:
            self.points = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    self.points.append((float(strs[1]), float(strs[2]), float(strs[3])))
                if strs[0] == "f":
                    pass
        self.points=np.array(self.points)
        self.number_of_points=self.points.shape[0]


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
        self.kdtree = o3d.geometry.KDTreeFlann(self.pc)

    def estimate_normals(self, max_k=16):
        '''
        Estimate normals considering a hybrid knn-neighborhood
        '''
        # TODO estimate normals from tangent plane fitted at vertex
        self.pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=max_k))
        self.pc = self.pc.normalize_normals()
        self.normals = np.asarray(self.pc.normals)
        assert self.normals.shape==(self.points.shape)

    def calculate_Qs(self, update_idx:list=None):
        # Tangent plane estimation for point sampled surfaces
        knn_idx = kneighbors_all(self.pc, k=5)
        
        # recompute for all vertices near the contraction
        if update_idx != None:
            update_group = []       # non-unique collection
            for vertex in update_idx:
                update_group.extend(knn_idx[vertex])

        for i, P in enumerate(knn_idx):
            if update_idx!=None and i not in update_group:
                continue    # skip all other points
            p, neighbors = P[0], P[1:]  # split into its own index and neighbors index
            assert p == i               # double check that this KNN return is in order
            planes = []
            Q = np.zeros((4,4))
            for pj in neighbors:
                ej = self.points[p] - self.points[pj]
                bj = np.cross(ej, self.pc.normals[p])
                # "tangent" plane on this edge is spanned by ej and bj
                tj = np.cross(ej, bj)
                tj = tj / np.linalg.norm(tj)
                # plane equation can be extracted from normal vector, since it satisfies a^2 + b^2 + c^2 = 1
                # and d can be set to 0 for convenience, let it pass through the origin
                plane = np.concatenate((tj, [0.0]))
                planes.append(plane)
                Q += np.outer(plane, plane) # This outer product is Kp for one plane

            self.Q_matrices[i]=Q
        pass


        ### OLD Face based plane computation
        ### Finds one plane equation for each face, thus different shape than finding one equation for each edge tangent
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
            ### OLD ###
        # Apply saliency weight here to each Q matrix
        # parameters alpha and lambda defined as in Lee et al. (2005)
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

    def apply_saliency_weight(self):
        assert len(self.Q_matrices) == len(self.points) and len(self.Q_matrices) == len(self.saliency), "Incorrect input dims"
        # Apply saliency weight here to each Q matrix
        # parameters alpha and lambda defined as in Lee et al. (2005)
        alpha = np.percentile(self.saliency, 30)
        print(f"The 30th percentile saliency is {alpha}")
        weight = np.where(self.saliency > alpha, self.lam*self.saliency, self.saliency)

        for i in range(len(self.Q_matrices)):
            self.Q_matrices[i] *= weight[i]
