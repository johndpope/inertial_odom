import tensorflow as tf
import math
import numpy as np 
from tensorflow.python.framework import ops
# from flow_transformer import transformer
import hyperparams as hyp
from PIL import Image
from scipy.misc import imsave
from math import pi
# from skimage.draw import *

def print_shape(t):
    print(t.name, t.get_shape().as_list())

def print_shape2(t, msg =''):
    def f(A):
        print np.shape(A), msg
        return A
    return tf.py_func(f, [t], t.dtype)

def split_rt(rt):
    shape = rt.get_shape()
    bs = int(shape[0])
    r = tf.slice(rt,[0,0,0],[-1,3,3])
    t = tf.reshape(tf.slice(rt,[0,0,3],[-1,3,1]),[bs,3])
    return r, t

def split_intrinsics(k):
    shape = k.get_shape()
    bs = int(shape[0])
    # fx = tf.slice(k,[0,0,0],[-1,1,1])
    # print_shape(fx)
    # fy = tf.slice(k,[0,1,0],[-1,1,1])
    # print_shape(fy)
    # x0 = tf.slice(k,[0,0,3],[-1,1,1])
    # print_shape(x0)
    # y0 = tf.slice(k,[0,1,3],[-1,1,1])
    # print_shape(y0)
    
    # fy = tf.reshape(tf.slice(k,[0,0,0],[-1,1,1]),[bs])
    # fx = tf.reshape(tf.slice(k,[0,1,1],[-1,1,1]),[bs])
    # y0 = tf.reshape(tf.slice(k,[0,0,2],[-1,1,1]),[bs])
    # x0 = tf.reshape(tf.slice(k,[0,1,2],[-1,1,1]),[bs])
    fx = tf.reshape(tf.slice(k,[0,0,0],[-1,1,1]),[bs])
    fy = tf.reshape(tf.slice(k,[0,1,1],[-1,1,1]),[bs])
    x0 = tf.reshape(tf.slice(k,[0,0,2],[-1,1,1]),[bs])
    y0 = tf.reshape(tf.slice(k,[0,1,2],[-1,1,1]),[bs])
    return fx,fy,x0,y0

def merge_rt(r,t):
    shape = r.get_shape()
    bs = int(shape[0])
    bottom_row = tf.tile(tf.reshape(tf.stack([0.,0.,0.,1.]),[1,1,4]),
                         [bs,1,1],name="bottom_row")
    rt = tf.concat(axis=2,values=[r,tf.expand_dims(t,2)],name="rt_3x4")
    rt = tf.concat(axis=1,values=[rt,bottom_row],name="rt_4x4")
    return rt

def random_crop(t,crop_h,crop_w,h,w):
    def off_h(): return tf.random_uniform([], minval=0, maxval=(h-crop_h-1), dtype=tf.int32)
    def off_w(): return tf.random_uniform([], minval=0, maxval=(w-crop_w-1), dtype=tf.int32)
    def zero(): return tf.constant(0)
    offset_h = tf.cond(tf.less(crop_h, h-1), off_h, zero)
    offset_w = tf.cond(tf.less(crop_w, w-1), off_w, zero)
    t_crop = tf.slice(t,[offset_h,offset_w,0],[crop_h,crop_w,-1],name="cropped_tensor")
    return t_crop, offset_h, offset_w

def near_topleft_crop(t,crop_h,crop_w,h,w,amount):
    ## take a random crop/pad somewhere in [-amount,amount]
    
    def get_rand(): return tf.random_uniform([], minval=0, maxval=amount, dtype=tf.int32)
    def get_zero(): return tf.constant(0)

    # pad a bit
    pad_h = tf.cond(tf.greater(amount, 0), get_rand, get_zero)
    pad_w = tf.cond(tf.greater(amount, 0), get_rand, get_zero)
    # t = tf.pad(t, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], "SYMMETRIC")
    t = tf.pad(t, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]])
    t = tf.slice(t,[0,0,0],[h,w,-1])
    
    # crop a bit
    offset_h = tf.cond(tf.less(crop_h, h), get_rand, get_zero)
    offset_w = tf.cond(tf.less(crop_w, w), get_rand, get_zero)
    t_crop = tf.slice(t,[offset_h,offset_w,0],[crop_h,crop_w,-1],name="cropped_tensor")
    return t_crop, offset_h, offset_w

def topleft_crop(t,crop_h,crop_w,h,w):
    offset_h = 0
    offset_w = 0
    t_crop = tf.slice(t,[offset_h,offset_w,0],[crop_h,crop_w,-1],name="cropped_tensor")
    return t_crop, offset_h, offset_w

def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    # return numpy.linalg.norm(transform[0:3,3])
    # t = tf.reshape(tf.slice(transform,[0,0,3],[-1,3,1]),[-1,3])
    t = tf.reshape(tf.slice(transform,[0,0,3],[-1,3,1]),[-1,3])
    # t should now be bs x 3  
    return tf.sqrt(tf.reduce_sum(tf.square(t),axis=1))


def compute_angle_3x3(R):
    return tf.acos(tf.minimum(1.,tf.maximum(-1.,(tf.trace(R)-1.)/2.)))

def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    # return numpy.arccos( min(1,max(-1, (numpy.trace(transform[0:3,0:3]) - 1)/2) ))
    r = tf.slice(transform,[0,0,0],[-1,3,3])
    return compute_angle_3x3(r)
    #return tf.acos(tf.minimum(1.,tf.maximum(-1.,(tf.trace(r)-1.)/2.)))

def compute_t_diff(rt1, rt2):
    """
    Compute the difference between the magnitudes of the translational components of the two transformations. 
    """
    t1 = tf.reshape(tf.slice(rt1,[0,0,3],[-1,3,1]),[-1,3])
    t2 = tf.reshape(tf.slice(rt2,[0,0,3],[-1,3,1]),[-1,3])
    # each t should now be bs x 3  
    mag_t1 = tf.sqrt(tf.reduce_sum(tf.square(t1),axis=1))
    mag_t2 = tf.sqrt(tf.reduce_sum(tf.square(t2),axis=1))
    return tf.abs(mag_t1-mag_t2)

def compute_t_ang(rt1, rt2):
    """
    Compute the angle between the translational components of two transformations.
    """
    t1 = tf.reshape(tf.slice(rt1,[0,0,3],[-1,3,1]),[-1,3])
    t2 = tf.reshape(tf.slice(rt2,[0,0,3],[-1,3,1]),[-1,3])
    # each t should now be bs x 3  
    mag_t1 = tf.sqrt(tf.reduce_sum(tf.square(t1),axis=1))
    mag_t2 = tf.sqrt(tf.reduce_sum(tf.square(t2),axis=1))
    dot = tf.reduce_sum(t1*t2,axis=1)
    return tf.acos(dot/(mag_t1*mag_t2 + hyp.eps))

def safe_inverse(a):
    """ 
    safe inverse for rigid transformations
    should be equivalent to 
      a_inv = tf.matrix_inverse(a)
    for well-behaved matrices
    """
    #shape = a.get_shape()
    #bs = int(shape[0])
    bs = tf.shape(a)[0]
    Ra = tf.slice(a,[0,0,0],[-1,3,3])
    Ta = tf.reshape(tf.slice(a,[0,0,3],[-1,3,1]),tf.stack([bs,3]))
    Ra_t = tf.transpose(Ra,[0,2,1])
    bottom_row = tf.tile(tf.reshape(tf.stack([0.,0.,0.,1.]),[1,1,4]),tf.stack([bs,1,1]))
    a_inv = tf.concat(axis=2,values=[Ra_t,-tf.matmul(Ra_t, tf.expand_dims(Ta,2))])
    a_inv = tf.concat(axis=1,values=[a_inv,bottom_row])
    return a_inv

def ominus(a,b):
    """
    Compute the relative 3D transformation between a and b.
    
    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)
    
    Output:
    Relative 3D transformation from a to b.
    https://github.com/liruihao/tools-for-rgbd-SLAM-evaluation/blob/master/evaluate_rpe.py
    """
    with tf.name_scope("ominus"):
        a_inv = safe_inverse(a)
        return tf.matmul(a_inv,b)

def ominus_ivn(a, b):
    """
    Compute the relative 3D transformation between a and b.
    
    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)
    
    Output:
    Relative 3D transformation from a to b.
    https://github.com/liruihao/tools-for-rgbd-SLAM-evaluation/blob/master/evaluate_rpe.py
    """

    with tf.name_scope("ominus_inv"):
        b_inv = safe_inverse(b)
        return tf.matmul(a,b_inv)

def sinabg2r(sina,sinb,sing):
    shape = sina.get_shape()
    one = tf.ones_like(sina,name="one")
    zero = tf.zeros_like(sina,name="one")
    cosa = tf.sqrt(1 - tf.square(sina))
    cosb = tf.sqrt(1 - tf.square(sinb))
    cosg = tf.sqrt(1 - tf.square(sing))
    Rz = tf.reshape(tf.stack([cosa, -sina, zero,
                             sina, cosa, zero,
                             zero, zero, one],
                    axis=1),[-1, 3, 3])
    Ry = tf.reshape(tf.stack([cosb, zero, sinb,
                             zero, one, zero,
                             -sinb, zero, cosb],
                    axis=1),[-1, 3, 3])
    Rx = tf.reshape(tf.stack([one, zero, zero,
                             zero, cosg, -sing,
                             zero, sing, cosg],
                    axis=1),[-1, 3, 3])
    Rcam=tf.matmul(tf.matmul(Rx,Ry),Rz,name="Rcam")
    return Rcam

def sinabg2r_fc(sina,sinb,sing):
    shape = sina.get_shape()
    bs = int(shape[0])
    hw = int(shape[1])
    one = tf.ones([bs,hw],name="one")
    zero = tf.zeros([bs,hw],name="zero")
    cosa = tf.sqrt(1 - tf.square(sina))
    cosb = tf.sqrt(1 - tf.square(sinb))
    cosg = tf.sqrt(1 - tf.square(sing))
    Rz = tf.reshape(tf.stack([cosa, -sina, zero,
                             sina, cosa, zero,
                             zero, zero, one],
                            axis=2),[bs, hw, 3, 3])
    Ry = tf.reshape(tf.stack([cosb, zero, sinb,
                             zero, one, zero,
                             -sinb, zero, cosb],
                            axis=2),[bs, hw, 3, 3])
    Rx = tf.reshape(tf.stack([one, zero, zero,
                             zero, cosg, -sing,
                             zero, sing, cosg],
                            axis=2),[bs, hw, 3, 3])

    Rcam=tf.matmul(tf.matmul(Rx,Ry),Rz,name="Rcam")
    
    # Rcam = tf.reshape(tf.pack([one, zero, zero,
    #                            zero, one, zero,
    #                            zero, zero, one],
    #                           axis=2),[bs, hw, 3, 3])
    return Rcam
    
def abg2r(a,b,g,bs):
    one = tf.ones([bs],name="one")
    zero = tf.zeros([bs],name="zero")
    sina = tf.sin(a)
    sinb = tf.sin(b)
    sing = tf.sin(g)
    cosa = tf.cos(a)
    cosb = tf.cos(b)
    cosg = tf.cos(g)
    Rz = tf.reshape(tf.stack([cosa, -sina, zero,
                             sina, cosa, zero,
                             zero, zero, one],
                    axis=1),[bs, 3, 3])
    Ry = tf.reshape(tf.stack([cosb, zero, sinb,
                             zero, one, zero,
                             -sinb, zero, cosb],
                    axis=1),[bs, 3, 3])
    Rx = tf.reshape(tf.stack([one, zero, zero,
                             zero, cosg, -sing,
                             zero, sing, cosg],
                    axis=1),[bs, 3, 3])
    Rcam=tf.matmul(tf.matmul(Rx,Ry),Rz,name="Rcam")
    return Rcam

def r2abg(r):
    # r is 3x3. i want to get out alpha, beta, and gamma
    # a = atan2(R(3,2), R(3,3));
    # b = atan2(-R(3,1), sqrt(R(3,2)*R(3,2) + R(3,3)*R(3,3)));
    # g = atan2(R(2,1), R(1,1));

    # x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
    # y = atan2(-R.at<double>(2,0), sy);
    # z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    
    r11 = r[:,0,0]
    r21 = r[:,1,0]
    r31 = r[:,2,0]
    r32 = r[:,2,1]
    r33 = r[:,2,2]
    a = atan2(r32,r33)
    b = atan2(-r31,tf.sqrt(r32*r32+r33*r33))
    g = atan2(r21,r11)
    return a, b, g

def zrt2flow_helper(Z1, rt12, fy, fx, y0, x0):
    r12, t12 = split_rt(rt12)
    # if hyp.dataset_name == 'KITTI' or hyp.dataset_name=='KITTI2':
    #     flow = zrt2flow_kitti(Z1, r12, t12, fy, fx, y0, x0)
    # else:
    flow = zrt2flow(Z1, r12, t12, fy, fx, y0, x0)
    return flow

def zrt2flow(Z, r, t, fy, fx, y0, x0):
    with tf.variable_scope("zrt2flow"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # get pointcloud1
        [grid_x1,grid_y1] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z,[bs,h,w],name="Z")
        XYZ = Camera2World(grid_x1,grid_y1,Z,fx,fy,x0,y0)

        # transform pointcloud1 using r and t, to estimate pointcloud2
        XYZ_transpose = tf.transpose(XYZ,perm=[0,2,1],name="XYZ_transpose")
        XYZ_mm = tf.matmul(r,XYZ_transpose,name="XYZ_mm")
        XYZ_rot = tf.transpose(XYZ_mm,perm=[0,2,1],name="XYZ_rot")
        t_tiled = tf.tile(tf.expand_dims(t,axis=1),[1,h*w,1],name="t_tiled")
        XYZ2 = tf.add(XYZ_rot,t_tiled,name="XYZ2")

        # project pointcloud2 down, so that we get the 2D location of all of these pixels
        [X2,Y2,Z2] = tf.split(axis=2, num_or_size_splits=3, value=XYZ2, name="splitXYZ")
        x2y2_flat = World2Camera(X2,Y2,Z2,fx,fy,x0,y0)
        [x2_flat,y2_flat]=tf.split(axis=2,num_or_size_splits=2,value=x2y2_flat,name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        x1_flat = tf.reshape(grid_x1,[bs,-1,1],name="x1")
        y1_flat = tf.reshape(grid_y1,[bs,-1,1],name="y1")
        flow_flat = tf.concat(axis=2,values=[x2_flat-x1_flat,y2_flat-y1_flat],name="flow_flat")
        flow = tf.reshape(flow_flat,[bs,h,w,2],name="flow")
        return flow

def Camera2World(x,y,Z,fx,fy,x0,y0):
    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])

    # the intrinsics are shaped [bs]
    # we need them to be shaped [bs,h,w]
    fy = tf.tile(tf.expand_dims(fy,1),[1,h*w])
    fx = tf.tile(tf.expand_dims(fx,1),[1,h*w])
    fy = tf.reshape(fy,[bs,h,w])
    fx = tf.reshape(fx,[bs,h,w])
    y0 = tf.tile(tf.expand_dims(y0,1),[1,h*w])
    x0 = tf.tile(tf.expand_dims(x0,1),[1,h*w])
    y0 = tf.reshape(y0,[bs,h,w])
    x0 = tf.reshape(x0,[bs,h,w])

    X=(Z/fx)*(x-x0)
    Y=(Z/fy)*(y-y0)
    pointcloud=tf.stack([tf.reshape(X,[bs,-1]),
                        tf.reshape(Y,[bs,-1]),
                        tf.reshape(Z,[bs,-1])],
                       axis=2,name="world_pointcloud")
    return pointcloud


def Camera2World_p(x,y,Z,fx,fy):
    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])

    # the intrinsics are shaped [bs]
    # we need them to be shaped [bs,h,w]
    fy = tf.tile(tf.expand_dims(fy,1),[1,h*w])
    fx = tf.tile(tf.expand_dims(fx,1),[1,h*w])
    fy = tf.reshape(fy,[bs,h,w])
    fx = tf.reshape(fx,[bs,h,w])
    
    X=(Z/fx)*x
    Y=(Z/fy)*y
    pointcloud=tf.stack([tf.reshape(X,[bs,-1]),
                        tf.reshape(Y,[bs,-1]),
                        tf.reshape(Z,[bs,-1])],
                       axis=2,name="world_pointcloud")
    return pointcloud


def World2Camera(X,Y,Z,fx,fy,x0,y0):
    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])

    # the intrinsics are shaped [bs]
    # we need them to be shaped [bs,h*w,1]
    fy = tf.tile(tf.expand_dims(fy,1),[1,h*w])
    fx = tf.tile(tf.expand_dims(fx,1),[1,h*w])
    y0 = tf.tile(tf.expand_dims(y0,1),[1,h*w])
    x0 = tf.tile(tf.expand_dims(x0,1),[1,h*w])
    fy = tf.reshape(fy,[bs,-1,1])
    fx = tf.reshape(fx,[bs,-1,1])
    y0 = tf.reshape(y0,[bs,-1,1])
    x0 = tf.reshape(x0,[bs,-1,1])
    
    x=(X*fx)/(Z+hyp.eps)+x0
    y=(Y*fy)/(Z+hyp.eps)+y0
    proj=tf.concat(axis=2,values=[x,y],name="camera_projection")
    return proj

def World2Camera_p(X,Y,Z,fx,fy):
    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])

    # the intrinsics are shaped [bs]
    # we need them to be shaped [bs,h*w,1]
    fy = tf.tile(tf.expand_dims(fy,1),[1,h*w])
    fx = tf.tile(tf.expand_dims(fx,1),[1,h*w])
    fy = tf.reshape(fy,[bs,-1,1])
    fx = tf.reshape(fx,[bs,-1,1])
    
    x=(X*fx)/(Z+hyp.eps)
    y=(Y*fy)/(Z+hyp.eps)
    proj=tf.concat(axis=2,values=[x,y],name="camera_projection")
    return proj

def atan2(y, x):
    with tf.variable_scope("atan2"):
        angle = tf.where(tf.greater(x, 0.0), tf.atan(y / x), tf.zeros_like(x))
        angle = tf.where(tf.greater(y, 0.0), 0.5 * np.pi - tf.atan(x / y), angle)
        angle = tf.where(tf.less(y, 0.0), -0.5 * np.pi - tf.atan(x / y), angle)
        angle = tf.where(tf.less(x, 0.0), tf.atan(y / x) + np.pi, angle)
        angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.equal(y, 0.0)),
                          np.nan * tf.zeros_like(x), angle)
        indices = tf.where(tf.less(angle, 0.0))
        updated_values = tf.gather_nd(angle, indices) + (2 * np.pi)
        update = tf.SparseTensor(indices, updated_values, angle.get_shape())
        update_dense = tf.sparse_tensor_to_dense(update)
        return angle + update_dense

def atan2_ocv(y, x):
    with tf.variable_scope("atan2_ocv"):
        # constants
        DBL_EPSILON = 2.2204460492503131e-16
        atan2_p1 = 0.9997878412794807 * (180 / np.pi)
        atan2_p3 = -0.3258083974640975 * (180 / np.pi)
        atan2_p5 = 0.1555786518463281 * (180 / np.pi)
        atan2_p7 = -0.04432655554792128 * (180 / np.pi)
        ax, ay = tf.abs(x), tf.abs(y)
        c = tf.where(tf.greater_equal(ax, ay), tf.div(ay, ax + DBL_EPSILON),
                      tf.div(ax, ay + DBL_EPSILON))
        c2 = tf.square(c)
        angle = (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c
        angle = tf.where(tf.greater_equal(ax, ay), angle, 90.0 - angle)
        angle = tf.where(tf.less(x, 0.0), 180.0 - angle, angle)
        angle = tf.where(tf.less(y, 0.0), 360.0 - angle, angle)
        return angle
                                                                                            
def normalize(tensor, a=0, b=1):
    with tf.variable_scope("normalize"):
        return tf.div(tf.multiply(tf.subtract(tensor, tf.reduce_min(tensor)), b - a),
                      tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))
    
def cart_to_polar_ocv(x, y, angle_in_degrees=False):
    with tf.variable_scope("cart_to_polar_ocv"):
        v = tf.sqrt(tf.add(tf.square(x), tf.square(y)))
        ang = atan2_ocv(y, x)
        scale = 1 if angle_in_degrees else np.pi / 180
        return v, tf.multiply(ang, scale)

def cart_to_polar(x, y, angle_in_degrees=False):
    with tf.variable_scope("cart_to_polar"):
        v = tf.sqrt(tf.add(tf.square(x), tf.square(y)))
        ang = atan2(y, x)
        scale = 180 / np.pi if angle_in_degrees else 1
        return v, tf.multiply(ang, scale)

def flow2color(flow):
    with tf.variable_scope("flow2color"):
        shape = flow.get_shape()
        bs, h, w, c = shape
        maxFlow = 40.0 #tf.maximum(20.0,tf.reduce_max(flow))
        # maxFlow = tf.maximum(20.0,tf.reduce_max(flow))
        # maxFlow = 5.0 #tf.maximum(20.0,tf.reduce_max(flow))
        # maxFlow = tf.maximum(5.0,tf.reduce_max(flow))
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[maxFlow*tf.ones([bs,h,1,1]),-maxFlow*tf.ones([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[-maxFlow*tf.ones([bs,h,1,1]),maxFlow*tf.ones([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[maxFlow*tf.ones([bs,h,1,1]),tf.zeros([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[-maxFlow*tf.ones([bs,h,1,1]),tf.zeros([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([bs,h,1,1]),maxFlow*tf.ones([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([bs,h,1,1]),-maxFlow*tf.ones([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([bs,h,1,1]),tf.zeros([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.zeros([bs,h,1,2]),flow])
        flow = tf.concat(axis=2,values=[maxFlow*tf.ones([bs,h,1,2]),flow])
        flow = tf.concat(axis=2,values=[-maxFlow*tf.ones([bs,h,1,2]),flow])
        fx, fy = flow[:, :, :, 0], flow[:, :, :, 1]
        fx = tf.clip_by_value(fx, -maxFlow, maxFlow)
        fy = tf.clip_by_value(fy, -maxFlow, maxFlow)
        v, ang = cart_to_polar_ocv(fx, fy)
        h = normalize(tf.multiply(ang, 180 / np.pi))
        s = tf.ones_like(h)
        v = normalize(v)
        hsv = tf.stack([h, s, v], 3)
        rgb = tf.image.hsv_to_rgb(hsv) * 255
        rgb = tf.slice(rgb,[0,0,10,0],[-1,-1,-1,-1])
        # rgb = rgb[0,0,1:,:]
        return tf.cast(rgb, tf.uint8)

def meshgrid2D(bs, height, width):
    with tf.variable_scope("meshgrid2D"):
        grid_x = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(0.0, width-1, width), 1), [1, 0]))
        grid_y = tf.matmul(tf.expand_dims(tf.linspace(0.0, height-1, height), 1),
                        tf.ones(shape=tf.stack([1, width])))
        grid_x = tf.tile(tf.expand_dims(grid_x,0),[bs,1,1],name="grid_x")
        grid_y = tf.tile(tf.expand_dims(grid_y,0),[bs,1,1],name="grid_y")
        return grid_x, grid_y
    
def warper(frame, flow, name="warper", is_train=True, reuse=False):
    with tf.variable_scope(name):
        shape = flow.get_shape()
        bs, h, w, c = shape
        if reuse:
            tf.get_variable_scope().reuse_variables()
        warp, occ = transformer(frame, flow, (int(h), int(w)))
        return warp, occ

def meshGridFlat(batchSize, height, width):
    with tf.name_scope('meshGridFlat'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #				 np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])

        baseGrid = tf.expand_dims(grid,0)
        grids = []
        for i in range(batchSize):
                grids.append(baseGrid)
        identityGrid = tf.concat(axis=0,values=grids)

        return identityGrid
    
def flowTransformGrid(flow):
    with tf.name_scope("flowTransformGrid"):
        flowShape = flow.get_shape()
        batchSize = flowShape[0]
        height = flowShape[1]
        width = flowShape[2]

        identityGrid = meshGridFlat(batchSize,height,width)

        flowU = tf.slice(flow,[0,0,0,0],[-1,-1,-1,1])
        flowV = tf.slice(flow,[0,0,0,1],[-1,-1,-1,1])

        #scale it to normalized range [-1,1]
        flowU = tf.reshape((flowU*2)/tf.cast(width, tf.float32),shape=tf.stack([batchSize,1,-1]))
        flowV = tf.reshape((flowV*2)/tf.cast(height, tf.float32),shape=tf.stack([batchSize,1,-1]))
        zeros = tf.zeros(shape=tf.stack([batchSize,1, height*width]))

        flowScaled = tf.concat(axis=1,values=[flowU,flowV,zeros])

        return identityGrid + flowScaled

def flowSamplingDensity(flow):
    def flatGridToDensity(u,v,h,w):
        with tf.name_scope("flatGridToDensity"):
            u = tf.squeeze(tf.clip_by_value(u,0,w-1))
            v = tf.squeeze(tf.clip_by_value(v,0,h-1))

            ids = tf.cast(u + (v*w),tf.int32)

            uniques,_,counts = tf.unique_with_counts(ids)
            densityMap = tf.sparse_to_dense(uniques,[h*w],counts,validate_indices=False)
            densityMap = tf.reshape(densityMap,[1,h,w])
            return densityMap

    with tf.name_scope("flowSamplingDensity"):
        flowShape = flow.get_shape()
        batchSize = flowShape[0].value
        h = flowShape[1].value
        w = flowShape[2].value

        grid = flowTransformGrid(flow)

        densities = []
        for it in range(0,batchSize):
            bGrid = tf.slice(grid,[it,0,0],[1,-1,-1])
            u = tf.cast(tf.floor((bGrid[:,0,:]+1)*w/2),tf.int32)
            v = tf.cast(tf.floor((bGrid[:,1,:]+1)*h/2),tf.int32)

            da = flatGridToDensity(u,v,h,w)
            #db = flatGridToDensity(u+1,v,h,w)
            #dc = flatGridToDensity(u,v+1,h,w)
            #dd = flatGridToDensity(u+1,v+1,h,w)
            densities.append(da)
            
        out = tf.concat(axis=0,values=densities)
        return out

def angleGrid(bs, h, w, y0, x0):
    grid_x, grid_y = meshgrid2D(bs, h, w)
    y0 = tf.tile(tf.reshape(y0,[bs,1,1]),[1,h,w])
    x0 = tf.tile(tf.reshape(x0,[bs,1,1]),[1,h,w])
    grid_y = grid_y - y0
    grid_x = grid_x - x0
    angleGrid = atan2_ocv(grid_y, grid_x)
    return angleGrid
    
def angle2color(angles):
    v = tf.ones_like(angles)
    h = normalize(tf.multiply(angles, 180 / np.pi))
    s = tf.ones_like(h)
    # v = normalize(v)
    hsv = tf.stack([h, s, v], 3)
    rgb = tf.image.hsv_to_rgb(hsv) * 255
    return tf.cast(rgb, tf.uint8)

def pseudoFlowColor(angles,depth):
    v = 1/depth
    h = normalize(tf.multiply(angles, 180 / np.pi))
    s = tf.ones_like(h)
    v = normalize(v)
    hsv = tf.stack([h, s, v], 3)
    rgb = tf.image.hsv_to_rgb(hsv) * 255
    return tf.cast(rgb, tf.uint8)

def resFlowColor(flow,angles,depth,tz):
    fx, fy = flow[:, :, :, 0], flow[:, :, :, 1]
    v, fang = cart_to_polar_ocv(fx, fy)
    # fang = atan2_ocv(fy,fx)
    # angles = fang-angles
    # angles = fang
    # v = 1/tf.square(fang-angles)
    # ax, ay = angles[:, :, :, 0], angles[:, :, :, 1]
    v = tz/depth
    # v = tf.ones_like(depth)
    h = normalize(tf.multiply(angles, 180 / np.pi))
    s = tf.ones_like(h)
    v = normalize(v)
    hsv = tf.stack([h, s, v], 3)
    rgb = tf.image.hsv_to_rgb(hsv) * 255
    return tf.cast(rgb, tf.uint8)

def oned2color(d,normalize=True):
    # convert a 1chan input to a 3chan image output
    # (it's not very colorful yet)
    if normalize:
        dmin = tf.reduce_min(d)
        dmax = tf.reduce_max(d)
    else:
        dmin = 0
        dmax = 1
    return tf.cast(tf.tile(255*((d-dmin)/(dmax-dmin)),[1,1,1,3]),tf.uint8)

def back2color(i):
    return tf.cast((i+0.5)*255,tf.uint8)

def zdrt2flow_fc(Z1, dp, R, T, scale, fy, fx):
    with tf.variable_scope("zdrt2flow_fc"):
        shape = Z1.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # x' = R(x-p) + p + t
        #    = R(x - (x + dp)) + (x + dp) + t # since p = (x + dp)
        #    = R(-dp) + x + dp + t
        
        # put the delta pivots into world coordinates
        [dpx, dpy, dpz] = tf.unstack(dp,axis=2)
        dpx = tf.reshape(dpx,[bs,h,w],name="dpx")
        dpy = tf.reshape(dpy,[bs,h,w],name="dpy")
        dpz = tf.reshape(dpz,[bs,h,w],name="dpz")
        XYZ_dp = Camera2World_p(dpx,dpy,dpz,fx,fy)

        # rotate the negative delta pivots
        XYZ_dp_rot = tf.matmul(R,-tf.expand_dims(XYZ_dp,3))
        XYZ_dp_rot = tf.reshape(XYZ_dp_rot,[bs,h*w,3])

        # create a pointcloud for the scene
        [grid_x1,grid_y1] = meshgrid2D(bs, h, w)
        Z1 = tf.reshape(Z1,[bs,h,w],name="Z1")
        XYZ = Camera2World_p(grid_x1,grid_y1,Z1,fx,fy)
        
        # add it all up
        XYZ_transformed = XYZ_dp_rot + XYZ + XYZ_dp + T
        
        # project down, so that we get the 2D location of all of these pixels
        [X2,Y2,Z2] = tf.split(axis=2, num_or_size_splits=3, value=XYZ_transformed, name="splitXYZ")
        x2y2_flat = World2Camera_p(X2,Y2,Z2,fx,fy)
        [x2_flat,y2_flat]=tf.split(axis=2,num_or_size_splits=2,value=x2y2_flat,name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        x1_flat = tf.reshape(grid_x1,[bs,-1,1],name="x1")
        y1_flat = tf.reshape(grid_y1,[bs,-1,1],name="y1")
        u = tf.reshape(x2_flat-x1_flat,[bs,h,w,1])
        v = tf.reshape(y2_flat-y1_flat,[bs,h,w,1])
        flow = tf.concat(axis=3,values=[u,v],name="flow")

        return flow, u, v, XYZ_transformed

def zcom2flow_fc(Z1, com1, com2, scale, fy, fx):
    with tf.variable_scope("zcom2flow_fc"):
        shape = Z1.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # assume that the motion is due to the center of mass changing place
        t = com2 - com1

        # create a pointcloud for the scene
        [grid_x1,grid_y1] = meshgrid2D(bs, h, w)
        Z1 = tf.reshape(Z1,[bs,h,w],name="Z1")
        XYZ = Camera2World_p(grid_x1,grid_y1,Z1,fx,fy)
        
        # add it up
        XYZ_transformed = XYZ + t
        
        # project down, so that we get the 2D location of all of these pixels
        [X2,Y2,Z2] = tf.split(axis=2, num_or_size_splits=3, value=XYZ_transformed, name="splitXYZ")
        x2y2_flat = World2Camera_p(X2,Y2,Z2,fx,fy)
        [x2_flat,y2_flat]=tf.split(axis=2,num_or_size_splits=2,value=x2y2_flat,name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        x1_flat = tf.reshape(grid_x1,[bs,-1,1],name="x1")
        y1_flat = tf.reshape(grid_y1,[bs,-1,1],name="y1")
        u = tf.reshape(x2_flat-x1_flat,[bs,h,w,1])
        v = tf.reshape(y2_flat-y1_flat,[bs,h,w,1])
        flow = tf.concat(axis=3,values=[u,v],name="flow")
        
        return flow, XYZ_transformed, x2y2_flat
    
def com2target(im, com):
    with tf.variable_scope("com2target"):
        [com_x,com_y] = tf.split(axis=2, num_or_size_splits=2, value=com, name="split_com")
        com_x = tf.slice(com,[0,0],[-1,1])
        com_y = tf.slice(com,[0,1],[-1,1])
        shape = im.get_shape()
        bs, h, w, c = shape
        im = tf.cast((i+0.5)*255,tf.uint8)
        im_left = tf.slice(im,[0,0,0,0],[-1,-1,-1,-1])
        im = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([bs,h,1,1]),tf.zeros([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([bs,h,1,1]),tf.zeros([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([bs,h,1,1]),tf.zeros([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([bs,h,1,1]),tf.zeros([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([bs,h,1,1]),tf.zeros([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([bs,h,1,1]),tf.zeros([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.concat(axis=3,values=[tf.zeros([bs,h,1,1]),tf.zeros([bs,h,1,1])]),flow])
        flow = tf.concat(axis=2,values=[tf.zeros([bs,h,1,2]),flow])
        flow = tf.concat(axis=2,values=[tf.zeros([bs,h,1,2]),flow])
        flow = tf.concat(axis=2,values=[-tf.zeros([bs,h,1,2]),flow])
        fx, fy = flow[:, :, :, 0], flow[:, :, :, 1]
        fx = tf.clip_by_value(fx, -maxFlow, maxFlow)
        fy = tf.clip_by_value(fy, -maxFlow, maxFlow)
        v, ang = cart_to_polar_ocv(fx, fy)
        h = normalize(tf.multiply(ang, 180 / np.pi))
        s = tf.ones_like(h)
        v = normalize(v)
        hsv = tf.stack([h, s, v], 3)
        rgb = tf.image.hsv_to_rgb(hsv) * 255
        rgb = tf.slice(rgb,[0,0,10,0],[-1,-1,-1,-1])
        # rgb = rgb[0,0,1:,:]
        return tf.cast(rgb, tf.uint8)
    
def zcom2com_fc(Z, com, fy, fx):
    with tf.variable_scope("zcom2com_fc"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # create a pointcloud for the scene
        [grid_x,grid_y] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z,[bs,h,w],name="Z")
        XYZ = Camera2World_p(grid_x,grid_y,Z,fx,fy)

        # move every point (on the toy, at least) to the com
        XYZ_transformed = XYZ + com
        
        # project down, so that we get the 2D location of all of these pixels
        [X2,Y2,Z2] = tf.split(axis=2, num_or_size_splits=3, value=XYZ_transformed, name="splitXYZ")
        x2y2_flat = World2Camera_p(X2,Y2,Z2,fx,fy)
        [x2_flat,y2_flat]=tf.split(axis=2,num_or_size_splits=2,value=x2y2_flat,name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        [grid_x1,grid_y1] = meshgrid2D(bs, h, w)
        x1_flat = tf.reshape(grid_x1,[bs,-1,1],name="x1")
        y1_flat = tf.reshape(grid_y1,[bs,-1,1],name="y1")
        u = tf.reshape(x2_flat-x1_flat,[bs,h,w,1])
        v = tf.reshape(y2_flat-y1_flat,[bs,h,w,1])
        flow = tf.concat(axis=3,values=[u,v],name="flow")
        return flow, XYZ_transformed

def offsetbbox(o1b, off_h, off_w, crop_h, crop_w, dotrim = False):
    o1b = tf.cast(o1b, tf.int32)
    #first get all 4 edges
    Ls = tf.slice(o1b, [0,0], [-1,1])
    Ts = tf.slice(o1b, [0,1], [-1,1])
    Rs = tf.slice(o1b, [0,2], [-1,1])
    Bs = tf.slice(o1b, [0,3], [-1,1])
    #next, offset by crop
    Ls = Ls - off_w
    Rs = Rs - off_w
    Ts = Ts - off_h
    Bs = Bs - off_h
    #finally, trim boxes if they go past edges
    if dotrim:
        assert False
        Ls = tf.maximum(Ls, 0)
        Rs = tf.maximum(Rs, 0)
        Ts = tf.maximum(Ts, 0)
        Bs = tf.maximum(Bs, 0)
        Ls = tf.minimum(Ls, crop_w)
        Rs = tf.minimum(Rs, crop_w)
        Ts = tf.minimum(Ts, crop_h)
        Bs = tf.minimum(Bs, crop_h)
    #then repack
    o1b = tf.concat(axis=1, values=[Ls, Ts, Rs, Bs]) 
    o1b = tf.cast(o1b, tf.int64)
    return o1b

#makes masks of 1's and 0's
def makemask(bbox2d):
    Ls = tf.slice(bbox2d, [0,0], [-1,1])
    Ts = tf.slice(bbox2d, [0,1], [-1,1])
    Rs = tf.slice(bbox2d, [0,2], [-1,1])
    Bs = tf.slice(bbox2d, [0,3], [-1,1])

    __f = lambda X: tf.cast(tf.tile(tf.expand_dims(X,2), [1, hyp.h, hyp.w]), tf.float32)
    Ls = __f(Ls)
    Ts = __f(Ts)
    Rs = __f(Rs)
    Bs = __f(Bs)

    x_t = tf.matmul(tf.ones(shape=tf.stack([hyp.h, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(0.0, hyp.w-1, hyp.w), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0, hyp.h-1, hyp.h), 1),
                    tf.ones(shape=tf.stack([1, hyp.w])))

    x_mask = tf.logical_and(x_t > Ls, x_t < Rs)
    y_mask = tf.logical_and(y_t > Ts, y_t < Bs)

    grid = tf.logical_and(x_mask, y_mask)

    return tf.cast(grid, tf.float32)

def avg2(a,b):
    return (a+b)/2.0

def makehalfmask(bbox2d):
    Ls = tf.slice(bbox2d, [0,0], [-1,1])
    Ts = tf.slice(bbox2d, [0,1], [-1,1])
    Rs = tf.slice(bbox2d, [0,2], [-1,1])
    Bs = tf.slice(bbox2d, [0,3], [-1,1])

    __f = lambda X: tf.cast(tf.tile(tf.expand_dims(X,2), [1, hyp.h, hyp.w]), tf.float32)
    Ls = __f(Ls)
    Ts = __f(Ts)
    Rs = __f(Rs)
    Bs = __f(Bs)

    Cx = avg2(Ls, Rs)
    Cy = avg2(Ts, Bs)
    Ls = avg2(Ls, Cx)
    Rs = avg2(Rs, Cx)
    Ts = avg2(Ts, Cy)
    Bs = avg2(Bs, Cy)

    x_t = tf.matmul(tf.ones(shape=tf.stack([hyp.h, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(0.0, hyp.w-1, hyp.w), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0, hyp.h-1, hyp.h), 1),
                    tf.ones(shape=tf.stack([1, hyp.w])))

    x_mask = tf.logical_and(x_t > Ls, x_t < Rs)
    y_mask = tf.logical_and(y_t > Ts, y_t < Bs)

    grid = tf.logical_and(x_mask, y_mask)

    return tf.cast(grid, tf.float32)


def decode_labels(mask, palette, num_images=1):
    n_classes = len(palette)
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img2 = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img2.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                assert 0 < k <= 14
                pixels[k_,j_] = palette[int(k)-1]
        outputs[i] = np.array(img2)
    return outputs

# we may want to easily change the way we encode/decode depth
# brox encodes it as 1/z
# eigen encodes it as log(z) 

def encode_depth_inv(x):
    return 1/(x+hyp.eps)
def decode_depth_inv(depth):
    return 1/(depth+hyp.eps)

def encode_depth_log(x):
    return tf.exp(x+hyp.eps)
def decode_depth_log(depth):
    return tf.log(depth+hyp.eps)

def encode_depth_id(x):
    return x
def decode_depth_id(depth):
    return depth

def encode_depth_invsig(x, delta = 0.01):
    return 1/(tf.nn.sigmoid(x)+delta)
def decode_depth_invsig(x, delta = 0.01):
    p = 1/(depth+hyp.eps)-delta
    p = tf.clip_by_value(p, hyp.eps, 1.0-hyp.eps)
    return -tf.log(1/p-1)

def encode_depth(x):
    if hyp.depth_encoding == 'id':
        return encode_depth_id(x)
    elif hyp.depth_encoding == 'log':
        return encode_depth_log(x)
    elif hyp.depth_encoding == 'inv':
        return encode_depth_inv(x)
    elif hyp.depth_encoding == 'invsig':
        return encode_depth_invsig(x)
    else:
        assert False

def decode_depth(depth):
    if hyp.depth_encoding == 'id':
        return decode_depth_id(depth)
    elif hyp.depth_encoding == 'log':
        return decode_depth_log(depth)
    elif hyp.depth_encoding == 'inv':
        return decode_depth_inv(depth)
    elif hyp.depth_encoding == 'invsig':
        return decode_depth_invsig(depth)
    else:
        assert False

def match(xs, ys): #sort of like a nested zip
    result = {}
    for i, (x,y) in enumerate(zip(xs, ys)):
        if type(x) == type([]):
            subresult = match(x, y)
            result.update(subresult)
        else:
            result[x] = y
    return result
    
def feed_from(inputs, variables, sess):
    return match(variables, sess.run(inputs))

def extract_gt_instance(O_things): #need some mapping here
    
    def f((num_obj, obj_id, bbox2d, pose3d, _)):
        classlabels = tf.slice(obj_id, [0,0], [-1,1])
        masks = makemask(bbox2d)
        masks = tf.cond(tf.greater(num_obj, 0), 
                        lambda: makemask(bbox2d), 
                        lambda: tf.zeros((0, hyp.h, hyp.w)))
        return num_obj, classlabels, bbox2d, pose3d, masks

    #if masks have height zero, and we stack them... we get in trouble
    n = tf.squeeze(tf.slice(O_things[0], [0], [1]))

    return tf.cond(tf.greater(n, 0), 
                   lambda: tf.map_fn(f, O_things, (tf.int64, tf.int64, tf.int64, tf.float32, tf.float32)),
                   lambda: [tf.zeros((hyp.bs,), dtype = tf.int64), 
                            tf.zeros((hyp.bs, 0, 1), dtype = tf.int64), 
                            tf.zeros((hyp.bs, 0, 4), dtype = tf.int64), 
                            tf.zeros((hyp.bs, 0, 3), dtype = tf.float32), 
                            tf.zeros((hyp.bs, 0, hyp.h, hyp.w), tf.float32)])

    #return tf.map_fn(f, O_things, (tf.int64, tf.int64, tf.int64, tf.float32, tf.float32))

def parse_gt_instance(O_things): #need some mapping here
    
    def f((num_obj, obj_id, bbox2d, pose9, occtrunc)):
        classlabels = tf.slice(obj_id, [0,0], [-1,1])
        position = tf.slice(pose9, [0,0], [-1,3])
        dimension = tf.slice(pose9, [0,3], [-1,3])
        pose3d = tf.slice(pose9, [0,6], [-1,3])
        return num_obj, classlabels, bbox2d, position, dimension, pose3d

    #if masks have height zero, and we stack them... we get in trouble
    n = tf.squeeze(tf.slice(O_things[0], [0], [1]))

    return tf.map_fn(f, O_things, (tf.int64, tf.int64, tf.int64, 
                                   tf.float32, tf.float32, tf.float32))

    '''
    return tf.cond(tf.greater(n, 0), 
                   lambda: tf.map_fn(f, O_things, (tf.int64, tf.int64, tf.int64, 
                                                   tf.float32, tf.float32, tf.float32)),
                   lambda: [tf.zeros((hyp.bs,), dtype = tf.int64), 
                            tf.zeros((hyp.bs, 0, 1), dtype = tf.int64), 
                            tf.zeros((hyp.bs, 0, 4), dtype = tf.int64), 
                            tf.zeros((hyp.bs, 0, 3), dtype = tf.float32), 
                            tf.zeros((hyp.bs, 0, 3), dtype = tf.float32), 
                            tf.zeros((hyp.bs, 0, 3), dtype = tf.float32)])
                            '''

def poses2rots_v2(poses):
    rx = tf.slice(poses, [0,0], [-1,1])
    ry = tf.slice(poses, [0,1], [-1,1])+math.pi/2.0
    rz = tf.slice(poses, [0,2], [-1,1])
    rots = sinabg2r(tf.sin(rz), tf.sin(ry), tf.sin(rx))
    return rots


def poses2rots(poses):
    rxryrz = tf.slice(poses, [0, 6], [-1, 3])
    rx = tf.slice(rxryrz, [0,0], [-1,1])
    ry = tf.slice(rxryrz, [0,1], [-1,1])+math.pi/2.0
    rz = tf.slice(rxryrz, [0,2], [-1,1])
    rots = sinabg2r(tf.sin(rz), tf.sin(ry), tf.sin(rx))
    return rots


def decompress_seg(num, seg):
    '''input: HxWx1, with max value being seg
    '''
    pass

def masks2boxes(masks):
    pass

def seg2masksandboxes(num, seg):
    #'gt_masks': 'masks of instances in this image. (instance-level masks), of shape (N, image_height, image_width)',
    #'gt_boxes': 'bounding boxes and classes of instances in this image, of shape (N, 5), each entry is (x1, y1, x2, y2)' 
    #the fifth feature of each box is the class id

    masks = decompress_seg(num, seg)
    boxes = masks2boxes(masks)
    
    masks = []
    boxes = []
    #assert False #not yet implemented
    return masks, boxes

def selectmask(cls_idx, masks):
    def select_single((idx, mask)):
        #idx is (), mask is 14x14x2
        return tf.slice(mask, tf.stack([0,0,idx]), [-1,-1,1])
    return tf.map_fn(select_single, [cls_idx, masks], 
                     parallel_iterations = 128, dtype = tf.float32)

def select_by_last(indices, items):
    axis = len(items.get_shape())
    def select_single((idx, item)):
        start = tf.stack([0 for i in range(axis-2)]+[idx])
        end = [-1 for i in range(axis-2)]+[1]
        return tf.squeeze(tf.slice(item, start, end))
    return tf.map_fn(select_single, [indices, items], 
                     parallel_iterations = 128, dtype = items.dtype)
        
def makeinfo(counts):
    def mi_single(count):
        im_info = tf.constant([[hyp.h, hyp.w, 1.0]])
        im_info = tf.tile(im_info, tf.stack([count, 1]))
        return im_info
    return tf.map_fn(mi_single, counts)

def mask2bbox(mask): #write this function
    #reduce on x and y
    yprofile = tf.reduce_any(mask, axis = 1) #reduce across x!
    xprofile = tf.reduce_any(mask, axis = 0) #y
    rangex = tf.range(0, hyp.w)
    rangey = tf.range(0, hyp.h)
    maskx = tf.boolean_mask(rangex, xprofile)
    masky = tf.boolean_mask(rangey, yprofile)
    minx = tf.reduce_min(maskx)
    maxx = tf.reduce_max(maskx)
    miny = tf.reduce_min(masky)
    maxy = tf.reduce_max(masky)
    #LTRB
    return tf.stack([minx, miny, maxx, maxy])

def extract_gt_boxes((counts, segs)):
    n = tf.squeeze(counts) #bs is 1
    seg = tf.squeeze(segs) #bs is 1

    def extract_instance(idx):
        mask = tf.equal(seg, idx+1)
        bbox = tf.concat(axis=0, values=[mask2bbox(mask), [0]])
        return bbox
    it = tf.range(0, n)
    result = tf.map_fn(extract_instance, it, dtype = tf.int32)
    return result

def vis_detections(im, class_name, dets, thresh=0.3):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    im = im[0,:,:,:]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print "%.2f confident that there is a %s at " % (score, class_name)
        print(bbox)
        y = int(bbox[0])
        x = int(bbox[1])

        y2 = int(bbox[2])
        x2 = int(bbox[3])
        im[x:x2,y,0] = 255
        im[x:x2,y,1] = 0
        im[x:x2,y,1] = 0
        im[x:x2,y2,0] = 255
        im[x:x2,y2,1] = 0
        im[x:x2,y2,1] = 0

        im[x,y:y2,0] = 255
        im[x,y:y2,1] = 0
        im[x,y:y2,2] = 0
        im[x2,y:y2,0] = 255
        im[x2,y:y2,1] = 0
        im[x2,y:y2,2] = 0
    return im #imsave(out_file, im)

def im2rim(im, SCALE):
    # convert an image to the format required by rcnn
    im_rgb = tf.cast(back2color(im),tf.float32)
    im_channels = tf.unstack(im_rgb, axis=-1)
    im_bgr = tf.stack([im_channels[2]-102.9801,
                       im_channels[1]-115.9465,
                       im_channels[0]-122.7717], axis=-1)
    
    return tf.image.resize_images(im_bgr, (hyp.h*SCALE, hyp.w*SCALE))
    #return im_bgr

def box_correspond_np(pred_boxes, gt_boxes, thresh = 0.0):
    #print 'box correspond'
    #print np.shape(pred_boxes)
    #print np.shape(gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(pred_boxes, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)) #Nx32
    dets_argmax_overlaps = overlaps.argmax(axis=1)
    dets_max_overlaps = overlaps.max(axis=1)
    matchmask = dets_max_overlaps > thresh

    return dets_argmax_overlaps, matchmask

def box_correspond(pred_boxes, gt_boxes, thresh = 0.75): 
    dets, mask = tf.py_func(box_correspond_np, [pred_boxes, gt_boxes, thresh], (tf.int64, tf.bool))
    #isempty = tf.greater(tf.shape(gt_boxes)[0], 0)
    dets = tf.reshape(dets, (tf.shape(pred_boxes)[0],))
    masks = tf.reshape(mask, (tf.shape(pred_boxes)[0],))
    return dets, masks

def rois_and_bboxdelta_to_bbox(bboxdelta, rois):
    boxes = rois[:, 1:5]
    assert len(np.shape(bboxdelta)) == 2
    bboxdelta = np.reshape(bboxdelta, [bboxdelta.shape[0], -1])
    imshape = [hyp.h, hyp.w, 3]
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        boxes = bbox_transform_inv(boxes, bboxdelta)
        boxes = _clip_boxes(boxes, imshape)
    else:
        boxes = np.tile(boxes, (1, probs.shape[1]))
    return boxes

def tf_rois_and_deltas_to_bbox(bboxdeltas, rois):
    assert cfg.TEST.BBOX_REG
    boxes = tf.slice(rois, [0, 1], [-1, 4])
    imshape = tf.constant([hyp.h, hyp.w, 3])
    # Apply bounding-box regression deltas 
    boxes = tf.py_func(bbox_transform_inv, [boxes, bboxdeltas], tf.float32)
    boxes = tf.py_func(_clip_boxes, [boxes, imshape], tf.float32)
    return tf.reshape(boxes, [-1, 84])

def tf_nms(dets, thresh):
    keep = tf.py_func(lambda a, b, c: np.array(nms(a,b,c)).astype(np.int64),
                      [dets, thresh, True],
                      tf.int64, stateful=False)
    return tf.cast(keep,tf.int32)


def get_good_detections(feats, probs, bboxdeltas, rois, many_classes=True, 
                        conf_thresh = 0.8, nms_thresh = 0.3):
    boxes = tf_rois_and_deltas_to_bbox(bboxdeltas, rois)
    if many_classes:
        cls_ind = 7
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_probs = probs[:, cls_ind]
    else:
        cls_boxes = boxes
        cls_probs = probs
        
    indices = tf.squeeze(tf.where(cls_probs > conf_thresh))
    
    feats = tf.gather(feats, indices)
    cls_probs = tf.gather(cls_probs, indices)
    cls_boxes = tf.gather(cls_boxes, indices)
    rois = tf.gather(rois, indices)

    # cls_boxes is [N x 4]
    # cls_probs is [N]; make it [N x 1] so we can concat
    cls_probs = tf.expand_dims(cls_probs,axis=-1)
    
    dets = tf.concat([cls_boxes, cls_probs],axis=1)
    keep = tf_nms(dets, nms_thresh)

    dets = tf.gather(dets, keep)
    feats = tf.gather(feats, keep)
    probs = tf.gather(probs, keep)
    deltas = tf.gather(bboxdeltas, keep)
    rois = tf.gather(rois, keep)
    
    # return dets, feats, probs, deltas
    # return dets, feats
    dets = tf.slice(dets, [0,0], [-1,4])
    return dets, feats

def pose_cls2angle(rxc, ryc, rzc):
    __f = lambda x: tf.cast(tf.argmax(x, axis = 1), tf.float32)
    xi = __f(rxc)
    yi = __f(ryc)
    zi = __f(rzc)
    rx = xi/36*2*pi - pi
    ry = yi/36*2*pi - pi
    rz = zi/36*2*pi - pi
    return tf.stack([rx, ry, rz], axis = 1)

def pose_angle2cls(rots):
    rx = rots[:,0]
    ry = rots[:,1]
    rz = rots[:,2]
    __round = lambda w: tf.cast(tf.round((w+pi)*36/(2*pi)), tf.int32)
    binx = __round(rx)
    biny = __round(ry)
    binz = __round(rz)
    #now make it categorical
    __oh = lambda w: tf.one_hot(w, depth = 36, axis = 1)
    clsx = __oh(binx)
    clsy = __oh(biny)
    clsz = __oh(binz)
    return clsx, clsy, clsz
    
def drawline(p1, p2, c, canvas, scale = 1):
    x1, y1 = p1.astype(np.int32)
    x2, y2 = p2.astype(np.int32)

    rr, cc, v = line_aa(y1, x1, y2, x2)
    stuff = np.stack([rr, cc, v], axis = 1)
    stuff = np.array([px for px in stuff if 
                      (0 <= px[0] < hyp.h*scale) and (0 <= px[1] < hyp.w*scale)], dtype = np.int32)
    if not stuff.size:
        return 

    rr = stuff[:,0]
    cc = stuff[:,1]
    v = stuff[:,2]

    assert np.max(canvas) <= 255.0
    assert np.min(canvas) >= 0.0
    
    canvas[rr, cc, c] = 255.0
    canvas[rr, cc, (c+1)%3] = 0.0
    canvas[rr, cc, (c+2)%3] = 0.0

def box2corners(bbox):
    L, T, R, B = bbox
    lb = np.array([L, B])
    lt = np.array([L, T])
    rb = np.array([R, B])
    rt = np.array([R, T])
    return [lb, lt, rb, rt]

def drawbox((lb, lt, rb, rt), c, canvas, scale = 1):
    drawline(lb, lt, c, canvas, scale)
    drawline(rb, rt, c, canvas, scale)
    drawline(lb, rb, c, canvas, scale)
    drawline(lt, rt, c, canvas, scale)
   
def drawbox2(bbox, canvas, c, scale = 1):
    corners = box2corners(bbox)
    drawbox([corner*scale for corner in corners], c, canvas, scale)

def plotcuboid2d(corners, canvas, c, scale = 1):
    for box in corners: 
        #there are twelve edges
        for i1, i2 in [(0,1),(0,2),(0,4),
                       (1,3),(1,5),(2,3),
                       (2,6),(3,7),(4,5),
                       (4,6),(5,7),(6,7)]:
            p1 = box[i1]*scale
            p2 = box[i2]*scale
            drawline(p1, p2, c, canvas, scale)
