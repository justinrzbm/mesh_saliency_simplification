# -*- coding: utf-8 -*-
"""
@author: Anton Wang
"""

import numpy as np
import sys
import open3d as o3d

from class_3d_model import a_3d_model
from knn import kneighbors_all

# Mesh simplification class
class mesh_simplify(a_3d_model):
    def __init__(self, input_filepath, threshold, simplify_ratio, saliency):
        if simplify_ratio>1 or simplify_ratio<=0:
            sys.exit('Error: simplification ratio should be (0<r<=1).')
        if threshold<0:
            sys.exit('Error: threshold should be (>=0).')
        super().__init__(input_filepath, saliency)
        print('Import model: '+str(input_filepath))
        self.t=threshold
        self.ratio=simplify_ratio
        self.saliency=saliency
        self.points_reduced = self.points[:]   # copy all starting points into shrinking set of points
        self.point_serial_number=np.arange(self.points.shape[0]) # to keep track of mapping

    # Select all valid pairs.
    def generate_valid_pairs(self):
        dist_pairs = []
        nn_idx = kneighbors_all(self.pc)
        k = len(nn_idx[0])
        for row in nn_idx:
            for j in range(1, k):
                dist_pairs.append((row[0], row[j]))
        dist_pairs = np.array(dist_pairs)

        # dist_pairs = []
        # for i in range(0, self.number_of_points):
        #     current_point_location=i+1
        #     current_point=self.points[i,:]
        #     current_point_to_others_dist=(np.sum((self.points-current_point)**2,axis=1))**0.5
        #     valid_pairs_location=np.where(current_point_to_others_dist<=self.t)[0]+1
        #     valid_pairs_location=valid_pairs_location.reshape(len(valid_pairs_location),1)
        #     current_valid_pairs=np.concatenate([current_point_location*np.ones((valid_pairs_location.shape[0],1)),valid_pairs_location],axis=1)
        #     if i==0:
        #         dist_pairs=current_valid_pairs
        #     else:
        #         dist_pairs=np.concatenate([dist_pairs, current_valid_pairs], axis=0)

        dist_pairs=np.array(dist_pairs)
        # loop removal
        find_same=dist_pairs[:,1]-dist_pairs[:,0]
        find_same_loc=np.where(find_same==0)[0]
        dist_pairs=np.delete(dist_pairs, find_same_loc, axis=0)
        
        self.valid_pairs = dist_pairs
        
        # duplicates are removed 
        unique_valid_pairs_trans, unique_valid_pairs_loc=np.unique(self.valid_pairs[:,0]*(10**10)+self.valid_pairs[:,1], return_index=True)
        self.valid_pairs=self.valid_pairs[unique_valid_pairs_loc,:]
    
    # Compute the optimal contraction target v_opt for each valid pair (v1, v2)
    # The error v_opt.T*(Q1+Q2)*v_pot of this target vertex becomes the cost of contracting that pair.
    # Place all the pairs in a heap keyed on cost with the minimum cost pair at the top
    def calculate_optimal_contraction_pairs_and_cost(self):
        self.v_optimal = []
        self.cost = []
        number_of_valid_pairs=self.valid_pairs.shape[0]
        for i in range(0, number_of_valid_pairs):
            current_valid_pair=self.valid_pairs[i,:]
            v_1_location=current_valid_pair[0]
            v_2_location=current_valid_pair[1]
            # find Q_1
            Q_1=self.Q_matrices[v_1_location]
            # find Q_2
            Q_2=self.Q_matrices[v_2_location]
            Q=Q_1+Q_2
            Q_new=np.concatenate([Q[:3,:], np.array([0,0,0,1]).reshape(1,4)], axis=0)
            if np.linalg.det(Q_new)>0:
                current_v_opt=np.matmul(np.linalg.inv(Q_new),np.array([0,0,0,1]).reshape(4,1))
                current_cost=np.matmul(np.matmul(current_v_opt.T, Q), current_v_opt)
                current_v_opt=current_v_opt.reshape(4)[:3]
            else:
                v_1=np.append(self.points[v_1_location,:],1).reshape(4,1)
                v_2=np.append(self.points[v_2_location,:],1).reshape(4,1)
                v_mid=(v_1+v_2)/2
                delta_v_1=np.matmul(np.matmul(v_1.T, Q), v_1)
                delta_v_2=np.matmul(np.matmul(v_2.T, Q), v_2)
                delta_v_mid=np.matmul(np.matmul(v_mid.T, Q), v_mid)
                current_cost=np.min(np.array([delta_v_1, delta_v_2, delta_v_mid]))
                min_delta_loc=np.argmin(np.array([delta_v_1, delta_v_2, delta_v_mid]))
                current_v_opt=np.concatenate([v_1,v_2,v_mid],axis=1)[:,min_delta_loc].reshape(4)
                current_v_opt=current_v_opt[:3]
            self.v_optimal.append(current_v_opt)
            self.cost.append(current_cost)
        
        self.v_optimal=np.array(self.v_optimal)
        self.cost=np.array(self.cost)
        self.cost=self.cost.reshape(self.cost.shape[0])
        
        cost_argsort=np.argsort(self.cost)
        self.valid_pairs=self.valid_pairs[cost_argsort,:]
        self.v_optimal=self.v_optimal[cost_argsort,:]
        self.cost=self.cost[cost_argsort]
        
        self.new_point=self.v_optimal[0,:]
        self.new_valid_pair=self.valid_pairs[0,:]
    
    # Iteratively remove the pair (v1, v2) of least cost from the heap
    # contract this pair, and update the costs of all valid pairs involving (v1, v2).
    # until existing points = ratio * original points
    def iteratively_remove_least_cost_valid_pairs(self):
        self.new_point_count=0
        while (self.number_of_points-self.new_point_count)>=self.ratio*(self.number_of_points):
            self.status_points=np.zeros(len(self.points))  # moved inside loop since deletion takes place every time
            
            # current valid pair
            current_valid_pair=self.new_valid_pair
            v_1_location=current_valid_pair[0] # point location in self.points (previously index shifted by 1 [was -1 to correct])
            v_2_location=current_valid_pair[1]

            # update self.points
            # put the top optimal vertex(point) into the sequence of points
            self.points[v_1_location,:]=self.new_point.reshape(1,3)
            self.points[v_2_location,:]=self.new_point.reshape(1,3)
            
            # set status of points
            # 0 means no change, -1 means the point is deleted
            # update v1, v2 to v_opt, then delete v2, keep v1
            self.status_points[v_2_location]=-1
                                    
            # update self.Q_matrices
            # OLD: self.update_Q(current_valid_pair-1, v_1_location)
            # Reinitialize point cloud object and kdtree for reduced point set, recalculate all normals ***This redundant operation is very slow in comparison, but o3d has no point cloud "removal"
            # self.points still contains the correctly indexed point set
            self.generate_new_pointcloud()
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(self.points_reduced)
            self.pc = pc

            self.estimate_normals(max_k=min(self.number_of_points//2, 16))
            self.calculate_Qs(update_idx=[v_1_location])
            
            # update self.valid_pairs, self.v_optimal, and self.cost
            self.update_valid_pairs_v_optimal_and_cost(v_1_location)
            # re-calculate optimal contraction pairs and cost
            self.update_optimal_contraction_pairs_and_cost(v_1_location)
            
            if self.new_point_count%100==0:
                print('Simplification: '+str(100*(self.number_of_points-self.new_point_count)/(self.number_of_points))+'%')
                print('Remaining: '+str(self.number_of_points-self.new_point_count)+' points')
                print('\n')
            
            self.new_point_count=self.new_point_count+1
            
        print('Simplification: '+str(100*(self.number_of_points-self.new_point_count)/(self.number_of_points+self.new_point_count))+'%')
        print('Remaining: '+str(self.number_of_points-self.new_point_count)+' points')
        print('End\n')
            
    # def update_Q(self, replace_locs, target_loc):
    #     # input: replace_locs, a numpy.array, shape: (2, ), locations of self.points need updating
    #     # input: target_loc, a number, location of self.points need updating
    #     face_set_index=np.where(self.faces==target_loc+1)[0]
    #     Q_temp=np.zeros((4,4))
        
    #     for j in face_set_index:
    #         p=self.plane_equ_para[j,:]
    #         p=p.reshape(1, len(p))
    #         Q_temp=Q_temp+np.matmul(p.T, p)
        
    #     for i in replace_locs:
    #         self.Q_matrices[i]=Q_temp
    
    def update_valid_pairs_v_optimal_and_cost(self, target_loc):
        # input: target_loc, a number, location of self.points need updating
        
        # processing self.valid_pairs        
        # replace all the point indexes containing current valid pair with new point index: target_loc+1
        v_1_loc_in_valid_pairs=np.where(self.valid_pairs==self.new_valid_pair[0])
        v_2_loc_in_valid_pairs=np.where(self.valid_pairs==self.new_valid_pair[1])
        
        self.valid_pairs[v_1_loc_in_valid_pairs]=target_loc
        self.valid_pairs[v_2_loc_in_valid_pairs]=target_loc
        
        delete_locs = []
        for item in v_1_loc_in_valid_pairs[0]:
            if np.where(v_2_loc_in_valid_pairs[0]==item)[0].size>0:
                delete_locs.append(item)
        delete_locs=np.array(delete_locs)
        
        find_same=self.valid_pairs[:,1]-self.valid_pairs[:,0]
        find_same_loc=np.where(find_same==0)[0]
        if find_same_loc.size >=1:
            delete_locs=np.append(delete_locs, find_same_loc)
        
        # delete process for self.valid_pairs, self.v_optimal and self.cost
        self.valid_pairs=np.delete(self.valid_pairs, delete_locs, axis=0)
        self.v_optimal=np.delete(self.v_optimal, delete_locs, axis=0)
        self.cost=np.delete(self.cost, delete_locs, axis=0)
        
        # unique process for self.valid_pairs, self.v_optimal and self.cost
        unique_valid_pairs_trans, unique_valid_pairs_loc=np.unique(self.valid_pairs[:,0]*(10**10)+self.valid_pairs[:,1], return_index=True)
        self.valid_pairs=self.valid_pairs[unique_valid_pairs_loc,:]
        self.v_optimal=self.v_optimal[unique_valid_pairs_loc,:]
        self.cost=self.cost[unique_valid_pairs_loc]
    
    def update_optimal_contraction_pairs_and_cost(self, target_loc):
        # input: target_loc, a number, location of self.points need updating
        v_target_loc_in_valid_pairs=np.where(self.valid_pairs==target_loc)[0]
        for i in v_target_loc_in_valid_pairs:
            current_valid_pair=self.valid_pairs[i,:]
            v_1_location=current_valid_pair[0]
            v_2_location=current_valid_pair[1]
            # find Q_1
            Q_1=self.Q_matrices[v_1_location]
            # find Q_2
            Q_2=self.Q_matrices[v_2_location]
            Q=Q_1+Q_2
            Q_new=np.concatenate([Q[:3,:], np.array([0,0,0,1]).reshape(1,4)], axis=0)
            if np.linalg.det(Q_new)>0:                
                current_v_opt=np.matmul(np.linalg.inv(Q_new),np.array([0,0,0,1]).reshape(4,1))
                current_cost=np.matmul(np.matmul(current_v_opt.T, Q), current_v_opt)
                current_v_opt=current_v_opt.reshape(4)[:3]
            else:
                v_1=np.append(self.points[v_1_location,:],1).reshape(4,1)
                v_2=np.append(self.points[v_2_location,:],1).reshape(4,1)
                v_mid=(v_1+v_2)/2
                delta_v_1=np.matmul(np.matmul(v_1.T, Q), v_1)
                delta_v_2=np.matmul(np.matmul(v_2.T, Q), v_2)
                delta_v_mid=np.matmul(np.matmul(v_mid.T, Q), v_mid)
                current_cost=np.min(np.array([delta_v_1, delta_v_2, delta_v_mid]))
                min_delta_loc=np.argmin(np.array([delta_v_1, delta_v_2, delta_v_mid]))
                current_v_opt=np.concatenate([v_1,v_2,v_mid],axis=1)[:,min_delta_loc].reshape(4)
                current_v_opt=current_v_opt[:3]
            self.v_optimal[i, :]=current_v_opt
            self.cost[i]=current_cost
        
        cost_argsort=np.argsort(self.cost)
        self.valid_pairs=self.valid_pairs[cost_argsort,:]
        self.v_optimal=self.v_optimal[cost_argsort,:]
        self.cost=self.cost[cost_argsort]
        
        self.new_point=self.v_optimal[0,:]
        self.new_valid_pair=self.valid_pairs[0,:]
    
    # Generate the simplified 3d pointcloud (points/vertices)
    def generate_new_pointcloud(self):
        # point_serial_number=np.arange(self.points_reduced.shape[0])   # moved to init
        points_to_delete_locs=np.where(self.status_points==-1)[0]
        serial_to_del=int(np.argwhere(self.point_serial_number==points_to_delete_locs))
        self.point_serial_number=np.delete(self.point_serial_number, serial_to_del)
        self.points_reduced=np.delete(self.points_reduced, serial_to_del, axis=0)
        self.Q_matrices = [self.Q_matrices[i] if i not in points_to_delete_locs else None for i in range(len(self.Q_matrices))]   # preserve indices
        # self.Q_matrices = [self.Q_matrices[i] for i in range(len(self.Q_matrices)) if i not in points_to_delete_locs]   # changes indices
        self.number_of_points=self.points_reduced.shape[0]
        # assert len(self.Q_matrices) == self.number_of_points
        assert len(self.point_serial_number) == self.number_of_points
    
    def output(self, output_filepath):
        np.save(output_filepath, self.points_reduced)
        print('Output simplified point cloud: '+str(output_filepath))