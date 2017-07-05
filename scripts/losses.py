import tensorflow as tf
from utils import *

def rtLoss(rt_e, rt_g):
    with tf.variable_scope("rtLoss"):
        shape = rt_e.get_shape()
        rt_eg = ominus(rt_e,rt_g)
        rtd = tf.reduce_mean(compute_distance(rt_eg))
        rta = tf.reduce_mean(compute_angle(rt_eg))
#        td = tf.reduce_mean(compute_t_diff(rt_e,rt_g))
#        ta = tf.reduce_mean(compute_t_ang(rt_e,rt_g))
        return rtd, rta

def l1Loss(e, g):
    with tf.variable_scope("l1Loss"):
        l = tf.reduce_mean(tf.abs(e - g))
        return l

def masked_l1Loss(e, g, valid):
    with tf.variable_scope("masked_l1Loss"):
        l = tf.reduce_mean(tf.abs(e-g),axis=3,keep_dims=True)
        l = l*valid
        l = tf.reduce_sum(l)/tf.reduce_sum(valid+hyp.eps)
        return l

def masked_hingeloss(e, g, valid, slack):
    with tf.variable_scope("masked_l1Loss"):
        diff = tf.abs(e-g)
        penalty = tf.where(diff > slack,
                           diff - slack,
                           tf.zeros_like(diff))
        l = tf.reduce_mean(penalty,axis=3,keep_dims=True)
        l = l*valid
        l = tf.reduce_sum(l)/tf.reduce_sum(valid+hyp.eps)
        return l

def l2Loss(e, g):
    with tf.variable_scope("l2Loss"):
        l = tf.reduce_mean(tf.square(e-g))
        return l
    
def masked_l2Loss(e, g, valid):
    with tf.variable_scope("l2Loss"):
        l = tf.reduce_mean(tf.square(e-g),axis=3,keep_dims=True)
        l = l*valid
        l = tf.reduce_sum(l)/tf.reduce_sum(valid+hyp.eps)
        return l
    
def scaleInvarLoss(e, g, lamb=0.5):
    with tf.variable_scope("scaleInvarLoss"):
        shape = e.get_shape()
        bs, h, w, c = shape
        d = e - g
        l2 = tf.reduce_mean(tf.square(d))
        n = h*w
        scale_invar = lamb*tf.square(tf.reduce_sum(d))/(n*n)
        return l2+scale_invar

def huberLoss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.reduce_mean(tf.where(condition, small_res, large_res))

def warped_z_loss(Z1_transformed,Z2,flow):
    # we want to check if Z2 and Z1_transformed are similar
    # but first we have to warp Z2 to the coordinate frame of Z1
    Z2 = tf.reshape(Z2,[hyp.bs,hyp.h,hyp.w,1])
    Z2_warped, _ = warper(Z2, flow)
    Z_diff = tf.abs(Z1_transformed-Z2_warped)
    xyzl = l1Loss(Z1_transformed, Z2_warped)
    return xyzl

        # [_,_,Z1_transformed] = tf.split(2, 3, XYZ1_transformed)
        # Z1_transformed = tf.reshape(Z1_transformed,[hyp.bs,hyp.h,hyp.w,1])
        # xyzl = warped_z_loss(Z1_transformed,Z2,flow_e)
        
def smoothLoss2(flow):
    with tf.name_scope("smoothLoss2"):
        shape = flow.get_shape()
        bs = shape[0]
        h = shape[1]
        w = shape[2]
        kernel = tf.transpose(tf.constant([[[[0,0,0],[0,1,-1],[0,0,0]]],
                                           [[[0,0,0],[0,1,0],[0,-1,0]]]],
                                          dtype=tf.float32),perm=[3,2,1,0],
                              name="kernel")
        [u,v] = tf.unstack(flow,axis=3)
        u = tf.expand_dims(u,3,name="u")
        v = tf.expand_dims(v,3,name="v")
        diff_u = tf.nn.conv2d(u,kernel,[1,1,1,1],padding="SAME",name="diff_u")
        diff_v = tf.nn.conv2d(v,kernel,[1,1,1,1],padding="SAME",name="diff_v")
        diffs = tf.concat(axis=3,values=[diff_u,diff_v],name="diffs")

        # make mask with ones everywhere but the bottom and right borders
        mask = tf.ones([bs,h-1,w-1,1],name="mask")
        mask = tf.concat(axis=1,values=[mask,tf.zeros([bs,1,w-1,1])],name="mask2")
        mask = tf.concat(axis=2,values=[mask,tf.zeros([bs,h,1,1])],name="mask3")
        loss = tf.reduce_mean(tf.abs(diffs*mask),name="loss")
        return loss

def smoothLoss1(u):
    with tf.name_scope("smoothLoss1"):
        shape = u.get_shape()
        bs = shape[0]
        h = shape[1]
        w = shape[2]
        kernel = tf.transpose(tf.constant([[[[0,0,0],[0,1,-1],[0,0,0]]],
                                           [[[0,0,0],[0,1,0],[0,-1,0]]]],
                                          dtype=tf.float32),perm=[3,2,1,0],
                              name="kernel")
        diff = tf.nn.conv2d(u,kernel,[1,1,1,1],padding="SAME",name="diff")

        # make mask with ones everywhere but the bottom and right borders
        mask = tf.ones([bs,h-1,w-1,1],name="mask")
        mask = tf.concat(axis=1,values=[mask,tf.zeros([bs,1,w-1,1])],name="mask2")
        mask = tf.concat(axis=2,values=[mask,tf.zeros([bs,h,1,1])],name="mask3")
        loss = tf.reduce_mean(tf.abs(diff*mask),name="loss")
        return loss

def masked_smoothLoss_multichan(u, valid):
    with tf.name_scope("smoothLoss1"):
        shape = u.get_shape()
        bs, h, w, d = shape
        kernel = tf.transpose(tf.constant([[[[0,0,0],[0,1,-1],[0,0,0]]],
                                           [[[0,0,0],[0,1,0],[0,-1,0]]]],
                                          dtype=tf.float32),perm=[3,2,1,0],
                              name="kernel")
        kernel = tf.tile(kernel, [1,1,int(d),1])
        diff = tf.nn.conv2d(u,kernel,[1,1,1,1],padding="SAME",name="diff")

        # make mask with ones everywhere but the bottom and right borders
        mask = tf.ones([bs,h-1,w-1,1],name="mask")
        mask = tf.concat(axis=1,values=[mask,tf.zeros([bs,1,w-1,1])],name="mask2")
        mask = tf.concat(axis=2,values=[mask,tf.zeros([bs,h,1,1])],name="mask3")
        mask = mask*valid
        loss = tf.reduce_sum(tf.abs(diff*mask))/tf.reduce_sum(mask+hyp.eps)
        return loss

def masked_smoothLoss1(u,valid):
    with tf.name_scope("smoothLoss1"):
        shape = u.get_shape()
        bs = shape[0]
        h = shape[1]
        w = shape[2]
        kernel = tf.transpose(tf.constant([[[[0,0,0],[0,1,-1],[0,0,0]]],
                                           [[[0,0,0],[0,1,0],[0,-1,0]]]],
                                          dtype=tf.float32),perm=[3,2,1,0],
                              name="kernel")
        diff = tf.nn.conv2d(u,kernel,[1,1,1,1],padding="SAME",name="diff")

        # make mask with ones everywhere but the bottom and right borders
        mask = tf.ones([bs,h-1,w-1,1],name="mask")
        mask = tf.concat(axis=1,values=[mask,tf.zeros([bs,1,w-1,1])],name="mask2")
        mask = tf.concat(axis=2,values=[mask,tf.zeros([bs,h,1,1])],name="mask3")
        mask = mask*valid
        loss = tf.reduce_sum(tf.abs(diff*mask))/tf.reduce_sum(mask+hyp.eps)
        return loss

