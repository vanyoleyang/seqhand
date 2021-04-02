import torch
import cv2
import imageio
import pickle
import os
from tqdm import tqdm

from numpy.linalg import norm

from Encoder_BH2MANO import Encoder_BH2MANO
from gen_3d_pose_flow import *

from manopth.manolayer import ManoLayer

from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints

from PIL import Image

def create_synth(verts, joints, skin_color, f, ss, tu, tv, rot, w, h, bg):
       rn = ColoredRenderer()
       R = cv2.Rodrigues(rot)[0]
       verts = np.transpose(np.matmul(R, np.transpose(verts)))
       joints = np.transpose(np.matmul(R, np.transpose(joints)))
       verts_3d = verts
       joints_3d = joints

       verts = np.array([[ss, ss, 1], ] * 778) * verts
       joints = np.array([[ss, ss, 1], ] * 21) * joints

       verts = verts + np.array([[tu, tv, 0], ] * 778)
       joints = joints + np.array([[tu, tv, 0], ] * 21)

       umax = np.max(verts[:, 0])
       umin = np.min(verts[:, 0])
       vmax = np.max(verts[:, 1])
       vmin = np.min(verts[:, 1])
       if ((umin < 0.) or (vmin < 0.) or (umax > w) or (vmax > h)):
              print('mesh outside')

       verts[:, 2] = 10. + (verts[:, 2] - np.mean(verts[:, 2]))
       verts[:, :2] = verts[:, :2] * np.expand_dims(verts[:, 2], 1)

       rn.camera = ProjectPoints(v=verts, rt=np.zeros(3), t=np.array([0, 0, 0]), f=np.array([1, 1]),
                                 c=np.array([0, 0]), k=np.zeros(5))

       rn.frustum = {'near': 1., 'far': 20., 'width': w, 'height': h}
       rn.set(v=verts, f=f, bgcolor=np.zeros(3))
       rn.vc = np.ones((778, 3))

       mask = rn.r.copy()
       mask = mask[:, :, 0].astype(np.uint8)

       rn.vc = skin_color
       hand = rn.r.copy() * 255.

       image = (1 - np.expand_dims(mask, 2)) * bg + np.expand_dims(mask, 2) * hand
       image = image.astype(np.uint8)

       image = Image.fromarray(image).resize((224, 224), Image.LANCZOS)
       return image, mask, verts_3d, joints_3d, verts, joints

def get_ss_tu_tv(verts, joints, w, h) :
       umax, umin, vmax, vmin = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])
       ss = w / ((random.uniform(1.7, 2.1) * np.max([umax - umin, vmax - vmin])))
       verts, joints = np.array([[ss, ss, 1], ] * 778) * verts, np.array([[ss, ss, 1], ] * 21) * joints
       umax, umin, vmax, vmin = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])

       hand_width = umax + umin
       hand_length = vmax + vmin
       tu = (w - hand_width) / 2. + umin + hand_width / 2. + random.uniform(15,15)
       tv = (h - hand_length) / 2. + vmin + hand_length / 2. + random.uniform(-50,45)

       rot_ind = random.randint(0,1)
       if rot_ind > 0.5 :
           angle = random.uniform(-np.pi, np.pi)
           axis = np.array([random.uniform(-1., 1.) for _ in range(3)])
           axis[random.randint(0, 2)] = 1.
           axis /= np.linalg.norm(axis)
           rot = angle * axis
       else : 
           angle = np.array([random.uniform(0.1, 0.3) for _ in range(3)])
           rot = np.array([-np.pi/3-angle[0], -np.pi/3-angle[1], np.pi/3+angle[2]])

       return ss, tu, tv, rot

def get_bg(bg_shape, size):
       bg_cent_x = random.randint(size / 2, bg_shape[0] - size / 2)
       bg_cent_y = random.randint(size / 2, bg_shape[1] - size / 2)
       return bg_cent_x, bg_cent_y

def get_hand_colors() :
       colors = []
       for i in range(0, 27):
              f = open('meshes_colored/%d.obj' % i)
              cont = f.readlines()
              f.close()
              col = []
              for x in cont:
                     a = x[:len(x) - 1].split(" ")
                     if (a[0] == 'v'):
                            col.append(np.array([float(a[4]), float(a[5]), float(a[6])]))
              col = np.expand_dims(np.vstack(col), 0)
              colors.append(col)
       return np.vstack(colors)

def main() :
       mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=6, flat_hand_mean=False, side='right')
       gpu = True
       num_flows = 2
       flow_length = 10

       pd = np.load('normalized_bh_inMANOorder.npy') ## Need to Download it
       model_path = 'BH2MANO_model/model_BH2MANO.pth'

       bg_path = 'BG_data/'
       bg_files = os.listdir(os.path.join(os.getcwd(), bg_path))

       data_dir = 'results/'

       enc_layer = Encoder_BH2MANO()
       enc_layer.load_state_dict(torch.load(model_path))
       if gpu:
              enc_layer.cuda()
              mano_layer.cuda()

       mano_layer.eval()
       enc_layer.eval()

       colors = get_hand_colors()

       gt = {}
       gt['joints_2d'] = []
       gt['cam_params'] = []
       gt['joints_3d'] = []
       gt['verts_3d'] = []
       for frame_idx in tqdm(range(num_flows)):

              pose_flow = gen_3d_pose_flow(pd.reshape((pd.shape[0], pd.shape[1] * pd.shape[2])), flow_length=flow_length)

              samples = pose_flow
              input_pose_joints_ = torch.tensor(samples).float().cuda()
              input_pose_joints = input_pose_joints_.view(input_pose_joints_.size(0), -1)
              pose_param,shape_param = enc_layer(input_pose_joints)
              pose_param[:, :3] *= 0.
              pose_param[:, 0] += np.pi
              shape_param = torch.rand(1, 10).expand(pose_param.size(0), -1).cuda().float() * 4. - 2.

              hand_verts, hand_joints = mano_layer(pose_param.clone(), shape_param.clone())
              hand_verts = hand_verts.cpu().detach().numpy() / 1000.
              hand_joints = hand_joints.cpu().detach().numpy() / 1000.

              f = mano_layer.th_faces.cpu().detach().numpy()

              color = colors[random.randint(0, 26)]
              size = 224
              w, h = size, size

              # flow_length = pose_flow.shape[0]
              assert flow_length == pose_flow.shape[0]

              ss, tu, tv, rot = get_ss_tu_tv(hand_verts[0], hand_joints[0], w, h)

              ss_end, tu_end, tv_end, rot_end = get_ss_tu_tv(hand_verts[-1], hand_joints[-1], w, h)
              rot_var_speed = random.uniform(0, 0.6) # random rotation speed

              ## Get Background
              while True :
                     bg_orig = imageio.imread(os.path.join(bg_path, random.choice(bg_files)))
                     if (bg_orig.shape[0] > size) and (bg_orig.shape[1] > size) : # get background that is large enough
                            break 
              bg_cent_x, bg_cent_y = get_bg(bg_orig.shape, size)
              bg_cent_end_x, bg_cent_end_y = get_bg(bg_orig.shape, size)
              bg = bg_orig[bg_cent_x - int(size / 2): bg_cent_x + int(size / 2),
                   bg_cent_y - int(size / 2): bg_cent_y + int(size / 2), :]

              ## Collect GTs
              images = []
              masks = []
              joints_2d = np.zeros((flow_length, 42))
              cam_params = np.zeros((flow_length, 27))
              joints_3d  = np.zeros((flow_length, 21, 3))
              verts_3d = np.zeros((flow_length, 778, 3))
              for i in range(flow_length):

                     img, mask, vert_3d, joint_3d, vert, joint = create_synth(hand_verts[i], hand_joints[i], color, f, ss, tu, tv, rot, w, h, bg)
                     images.append(img)
                     masks.append(mask * 255)
                     cam_params[i, :] = np.concatenate([np.array([1., ss, tu, tv]), rot,
                                                        pose_param[i, 3:].detach().cpu().numpy(),
                                                        shape_param[i].detach().cpu().numpy()], 0)
                     joints_2d[i, :] = joint[:,:2].reshape((42))
                     joints_3d[i, :, :] = joint_3d
                     verts_3d[i, :, :] = vert_3d

                     ss = ss + (ss_end - ss) / flow_length * 0.5
                     tu = tu + (tu_end - tu) / flow_length * 0.2
                     tv = tv + (tv_end - tv) / flow_length * 0.2
                     rot = rot + (rot_end - rot) / flow_length * rot_var_speed
                     bg_cent_x = int(bg_cent_x + (bg_cent_end_x - bg_cent_x) / flow_length)
                     bg_cent_y = int(bg_cent_y + (bg_cent_end_y - bg_cent_y) / flow_length)
                     bg = bg_orig[bg_cent_x - int(size / 2):bg_cent_x + int(size / 2),
                          bg_cent_y - int(size / 2):bg_cent_y + int(size / 2), :]

              gt['joints_2d'].append(joints_2d)
              gt['joints_3d'].append(joints_3d)
              gt['verts_3d'].append(verts_3d)
              gt['cam_params'].append(cam_params)
              imageio.mimsave(data_dir + 'gifs/%s.gif' % (frame_idx), images)
              imageio.mimsave(data_dir + 'masks/synth_%s_mask.gif' % (frame_idx), masks)
       with open(data_dir + 'ground_truths.pickle', 'wb') as fo:
              pickle.dump(gt, fo, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    main()


