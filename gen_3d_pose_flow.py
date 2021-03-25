import numpy as np
import random

def gen_3d_pose_flow(pd, flow_length=50, alpha=1.2) :
       flow_poses = np.zeros((flow_length, pd.shape[1]))

       ### set initial and ending (random) hand pose
       starting_pose_ind, end_pose_ind = random.randint(0, pd.shape[0]-1), random.randint(0, pd.shape[0] - 1)

       pose_curr = pd[starting_pose_ind]
       end_pose = pd[end_pose_ind]
       flow_poses[0, :] = pose_curr

       pose_prev = None

       for i in list(range(1, flow_length)) :
              pose_curr -= (alpha / flow_length) * (pose_curr - end_pose)
              selected_pose = get_k_th_nearest_pose(pd, pose_curr, pose_prev, i)
              flow_poses[i, :] = selected_pose
              pose_prev = pose_curr
              pose_curr = selected_pose

       return flow_poses

def get_k_th_nearest_pose(pd, pose_curr, pose_prev, step, k =0) :
       pose_curr_ = np.repeat(pose_curr[np.newaxis, :], pd.shape[0], axis=0)

       euc_sim = np.sum(np.abs(pose_curr_ - pd), axis=1)
       if step > 1 :
              euc_sim += np.sum(np.abs(pose_prev - pd), axis=1)
       candidate_indices = np.argsort(euc_sim)

       return pd[candidate_indices[k]]

