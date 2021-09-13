"""
Applying HMR to Handtool videos.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px.
Realtime-Pose [Cao et al 17] output is required to figure out the bbox and the right scale factor.

Sample usage:
img_dir=/Users/zoli/Work/data/Ring-Fit-poses/frames
vis_dir=/Users/zoli/Work/data/Ring-Fit-poses/HMR
openpose_path=/Users/zoli/Work/data/Ring-Fit-poses/Openpose-video/Openpose-video.pkl
save_path=/Users/zoli/Work/data/Ring-Fit-poses/HMR/HMR.pkl
echo python run_imagefolder.py --image_folder ${img_dir} --vis_folder ${vis_dir} --openpose_path ${openpose_path} --save_path ${save_path} --save_after_each_iteration True
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import h5py
import numpy as np
import cPickle as pk
from absl import flags
from glob import glob
import matplotlib
matplotlib.use('Agg')
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel


flags.DEFINE_string('image_folder', '/path/to/image/folder', 'path to imagefolder')
flags.DEFINE_string('vis_folder', '/path/to/vis/folder', 'path to visualization folder')
flags.DEFINE_string('openpose_path', '/path/to/Openpose-video.pkl', 'Openpose path')
flags.DEFINE_string('save_path', '/path/to/pkl/file', 'save path')
#flags.DEFINE_boolean('run_bbox_only', False, 'save bounding box and stop execution')
flags.DEFINE_boolean('save_after_each_iteration', False, 'save after each iteration')



def visualize(img, proc_param, joints, verts, cam, save_path=None):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig, trans = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])
    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1, figsize=(16,12))
    plt.clf()
    plt.subplot(221)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    #plt.show()
    # import ipdb
    # ipdb.set_trace()
    if save_path is not None:
        plt.savefig(save_path)
    return cam_for_render, trans, joints_orig, vert_shifted


def get_bbox_realtime_pose(j2d_pose, vis_thr=0.2, min_person_height=60.):
    '''
    use different bbox for different frames.
    if person_height is too small, i.e, there are too few detected joints,
    we keep a minimum height for compting the scaling factor
    '''
    kp = j2d_pose
    vis = kp[:, 2] > vis_thr
    num_detected_joints = np.sum(vis.astype(int))
    if num_detected_joints > 6:
        vis_kp = kp[vis, :2]
        min_pt = np.min(vis_kp, axis=0)
        max_pt = np.max(vis_kp, axis=0)
        person_height = np.linalg.norm(max_pt - min_pt)
        if person_height < min_person_height:
            person_height = min_person_height
        # if person_height == 0:
        #     print('bad!')
        #     import ipdb
        #     ipdb.set_trace()
        center = (min_pt + max_pt) / 2.
        scale = 150. / person_height
    else:
        scale = 1.
        center = None
    return scale, center


def preprocess_image(img_path, j2d_pose, crop_path=None):
    # use different bbox for different frames
    img = io.imread(img_path)
    # get bounding box from Openpose output
    scale, center = get_bbox_realtime_pose(j2d_pose)
    # use image center instead if the center is very closed to image border
    height, width, _ = img.shape

    # padding = 20.
    # if center is not None:
    #     if center[0]<padding or center[0]>height-padding:
    #         center[0] = height/2.
    #     if center[1]<padding or center[1]>width-padding:
    #         center[1] = width/2.
    # else:
    #     center = np.array([height/2., width/2.])
    if center is None:
        center = np.array([height/2., width/2.])

    # scale and crop
    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)
    # saving image patch containing target person
    if crop_path is not None:
        io.imsave(crop_path, crop)
    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)
    return scale, center, crop, proc_param, img


if __name__ == '__main__':

    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1

    image_folder = config.image_folder
    vis_folder = config.vis_folder
    openpose_path = config.openpose_path
    save_path = config.save_path
    #run_bbox_only = config.run_bbox_only
    save_after_each_iteration = config.save_after_each_iteration

    print("****** inputs ******")
    print("image_folder: {}".format(image_folder))
    print("vis_folder: {}".format(vis_folder))
    print("openpose_path: {}".format(openpose_path))
    print("save_path: {}".format(save_path))
    #print("run_bbox_only: {}".format(run_bbox_only))

    # ------------------------------------------------------------
    # Load useful information

    print("Loading image folder info from {0:s} ...".format(
        join(image_folder, "data_info.pkl")))
    with open(join(image_folder, "data_info.pkl"), 'r') as f:
        data_info = pk.load(f)
        image_names = data_info["image_names"]
        item_names = data_info["item_names"]
        item_lengths = data_info["item_lengths"]
        item_to_image = data_info["item_to_image"]
        #image_to_itemframe = data_info["image_to_itemframe"]


    # Check item_start and item_end
    item_start = 1
    item_end = -1
    nitems = len(item_names)
    if not 1<=item_start<=nitems:
        print("Check failed: 1<=item_start<=nitems (1<={0:d}<={1:d})".format(item_start, nitems))
    if item_end == -1:
        item_end = nitems
    elif not 1<=item_end<=nitems:
        print("Check failed: 1<=item_end<=nitems (1<={0:d}<={1:d})".format(item_end, nitems))

    if not exists(dirname(save_path)):
        makedirs(dirname(save_path))


    # Load Openpose-video results
    with open(openpose_path, 'r') as f_j2d:
        data_j2d = pk.load(f_j2d)


    # ------------------------------------------------------------
    # Run HMR on each of the items (i.e., video frames and
    # individual images) within image_folder
    if not exists(vis_folder):
        makedirs(vis_folder)


    results_dict = dict()

    # load pretrained model
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    for i in range(item_start-1, item_end):
        item_name = item_names[i]
        print("  Running HMR on item #{0:d}: {1:s} ...".format(i, item_name))
        nframes = item_lengths[i]

        # Get images_paths from data_info
        image_paths = [None]*nframes
        for k in range(nframes):
            image_paths[k] = join(image_folder,
                                  image_names[item_to_image[i] + k])

        if nframes > 1: # video frames
            vis_dir = join(vis_folder, item_name)
        else: # individual images
            vis_dir = vis_folder

        # load 2D poses
        j2d_pose = data_j2d[item_name]
        if j2d_pose.shape[0] != nframes:
            raise ValueError('j2d_pose.shape[0] != nframes')

        # SMPL renderer
        renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

        pred_thetas = np.zeros((nframes, 85))
        pred_cams = np.zeros((nframes, 3))
        pred_trans = np.zeros((nframes, 3))
        pred_poses = np.zeros((nframes, 72))
        pred_shapes = np.zeros((nframes, 10))
        joint_2d_positions = np.zeros((nframes, 19, 2))
        #bbox_fids = np.zeros(nframes).astype(int)
        #bbox_scales = np.zeros(nframes)
        #bbox_centers = np.zeros((nframes, 2))

        vis_dir = join(vis_folder, item_name)
        bbox_vis_dir = join(vis_folder, 'bbox_{}'.format(item_name))
        if not exists(vis_dir):
            makedirs(vis_dir)
        if not exists(bbox_vis_dir):
            makedirs(bbox_vis_dir)

        # # use same bbox for all frames, using Openpose outputs
        # scale, center = get_bbox_realtime_pose(j2d_pose)
        # # correct center in extreme cases
        # img_example = io.imread(image_paths[0])
        # frame_height, frame_width, _ = img_example.shape
        # padding = 20.
        # if center[0]<padding or center[0]>frame_height-padding:
        #     center[0] = frame_height/2.
        # if center[1]<padding or center[1]>frame_width-padding:
        #     center[1] = frame_width/2.

        for n in range(nframes):
            print("  frame #{0}".format(n))
            img_path = image_paths[n]
            image_name = basename(img_path)
            crop_path = join(bbox_vis_dir, image_name)
            # use different bbox for different frames
            scale, center, input_img, proc_param, img = preprocess_image(
               img_path, j2d_pose[n], crop_path)
            # # use same bbox for all frames
            # input_img, proc_param, img = preprocess_image(img_path, scale, center, crop_path)

            #bbox_fids[n] = n
            #bbox_scales[n] = scale
            #bbox_centers[n] = center.copy()

            # Add batch dimension: 1 x D x D x 3
            input_img = np.expand_dims(input_img, 0)

            joints, verts, cams, joints3d, theta = model.predict(
                input_img, get_theta=True)

            # Visualization
            vis_path = join(vis_dir, image_name)
            cam_for_render, trans, joints_orig, vert_shifted = \
                visualize(img, proc_param, joints[0], verts[0],
                          cams[0], vis_path)

            theta = theta.reshape(-1)
            pred_thetas[n] = theta.copy()
            pred_cams[n] = cam_for_render.copy() # note that theta[:3] is the cam for cropped image 224 x 224
            pred_trans[n] = trans.copy()
            pred_poses[n] = theta[3:75].copy()
            pred_shapes[n] = theta[75:].copy()
            joint_2d_positions[n] = joints_orig.copy()


        # if run_bbox_only:
        #     # saving bbox in hdf5 format
        #     h5_path = join(save_dir, 'bbox_2d.h5')
        #     if exists(h5_path):
        #         remove(h5_path)
        #         print("old file removed: {}".format(h5_path))
        #     fh5 = h5py.File(h5_path, "w")
        #     fh5.create_dataset("fids", dtype='i', data=bbox_fids)
        #     fh5.create_dataset("scales", dtype='f', data=bbox_scales)
        #     fh5.create_dataset("centers", dtype='f', data=bbox_centers)

        # Save estimated 3D poses
        results = {'thetas': pred_thetas,
                   'cams': pred_cams,
                   'trans': pred_trans,
                   'poses': pred_poses,
                   'shapes': pred_shapes,
                   'joint_2d_positions': joint_2d_positions}
        results_dict[item_name] = results

        if save_after_each_iteration:
            if exists(save_path):
                with open(save_path, 'r') as f:
                    data = pk.load(f)
                    data[item_name] = results
            else:
                data = {item_name: results}

            with open(save_path, 'w') as f:
                pk.dump(data, f)


    if not save_after_each_iteration:
        with open(save_path, 'w') as f:
            pk.dump(results_dict, f)
