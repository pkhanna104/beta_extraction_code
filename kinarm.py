""" Module for common kinarm-related tasks
"""
import numpy as np
#import utils_.geometry as geometry

# # Jeev: 082712 dataset: 
# sho_x =0.0509
# sho_y =-0.1413
# L1 =0.1493
# L2 =0.2437
# L2ptr =0.0293

# calib = kinarm.calib(sho_x, shoy, L1, L2, L2ptr)
# dat = np.vstack((x['AD33'], x['AD34'], x['AD35'], x['AD36'])).T, want to be a T x 4 array
# xpos,ypos,xvel,yvel = kinarm.calc_endpt(calib, dat[:,0], dat[:,1], sh_vel = dat[:,2], el_vel = dat[:,3])

class calib():
    def __init__(self, sho_x, sho_y, L1, L2, L2ptr):
        self.sho_x = sho_x
        self.sho_y = sho_y
        self.L1 = L1
        self.L2 = L2
        self.L2ptr = L2ptr

def load_calib_file(fname, arm='RIGHT'):
    """ Load calibration file"""
    if arm not in ['RIGHT', 'LEFT']:
        raise Exception("kinarm.load_calib_file: arm type not recongized: %s" % arm)

    import scipy.io
    f = scipy.io.loadmat(fname)
    sho_x = f['c3d_data'][0,0]['CALIBRATION_%s_SHO_X' % arm][0,0]
    sho_y = f['c3d_data'][0,0]['CALIBRATION_%s_SHO_Y' % arm][0,0] 
    L1 = f['c3d_data'][0,0]['CALIBRATION_%s_L1' % arm][0,0] 
    L2 = f['c3d_data'][0,0]['CALIBRATION_%s_L2' % arm][0,0] 
    L2ptr = f['c3d_data'][0,0]['CALIBRATION_%s_PTR_ANTERIOR' % arm][0,0] 

    return calib(sho_x, sho_y, L1, L2, L2ptr)

def convert_pos(screen_pos, L1, L2, L2ptr, sho_pos_x, sho_pos_y):
    """ Target coordinates from DexteritE specified relative to hand 
    calibration point 
    """
    center_screen = 0.01*np.array(screen_pos)  #convert to meters
    
    # joint angles used for computing (0,0) in DexteritE 
    # all targets are specified in coordinates relative to this point
    thetaE_cal = 90*np.pi/180
    thetaS_cal = 30*np.pi/180
    
    #convert this point to hand position via Jacobian
    elb_pos_x, elb_pos_y = pol2cart(thetaS_cal, L1)
    hand_pos_x, hand_pos_y = pol2cart(thetaE_cal+thetaS_cal, L2)
    pointer_x, pointer_y = pol2cart(thetaE_cal+thetaS_cal+np.pi/2, L2ptr)
    hand_pos_x += elb_pos_x + pointer_x + sho_pos_x
    hand_pos_y += elb_pos_y + pointer_y + sho_pos_y
    
    #shift target positions to absolute coordinates
    hand_pos = np.array([hand_pos_x, hand_pos_y]).reshape((2,1))
    center_hand = center_screen + np.tile(hand_pos, [1, center_screen.shape[1]])

    return center_hand

def calc_endpt(calib, sh_pos, el_pos, sh_vel=None, el_vel=None,
    sh_acc=None, el_acc=None, arm='right'):
    """
    Convert from joint coordinates to endpoint coordinates, 
    based on joint2endpt.m

    Assume inputs are AD channels: 
    """
    joint_angle_gain = 2.5
    joint_angular_vel_gain = 0.5
    joint_angular_acc_gain = 0.01
    joint_angle_offset = 4./joint_angle_gain
    V_per_level = 5./2**11
    
    sh_pos = sh_pos/joint_angle_gain + joint_angle_offset
    el_pos = el_pos/joint_angle_gain + joint_angle_offset

    sh_vel = sh_vel/joint_angular_vel_gain
    el_vel = el_vel/joint_angular_vel_gain
    
    L1 = calib.L1
    L2 = calib.L2
    L2ptr = calib.L2ptr
    sho_x = calib.sho_x
    sho_y = calib.sho_y

    sum_angle = sh_pos + el_pos
    sum_angle_orth = sh_pos + el_pos + np.pi/2

    if arm == 'left':
        sh_pos = np.pi - sh_pos
        sum_angle = np.pi - sum_angle
        sum_angle_orth = np.pi - sum_angle_orth

    elb_pos_x, elb_pos_y = pol2cart(sh_pos, calib.L1)
    hand_pos_x, hand_pos_y = pol2cart(sum_angle, calib.L2)
    pointer_x, pointer_y = pol2cart(sum_angle_orth, calib.L2ptr)

    x_pos = hand_pos_x + elb_pos_x + pointer_x + calib.sho_x
    y_pos = hand_pos_y + elb_pos_y + pointer_y + calib.sho_y

    if not (sh_vel == None or el_vel == None) and arm == 'right':
        sin_angle = np.sin(sh_pos)
        sin_sum_angle = np.sin(sum_angle)
        sin_sum_angle_orth = np.sin(sum_angle_orth)
        cos_angle = np.cos(sh_pos)
        cos_sum_angle = np.cos(sum_angle)
        cos_sum_angle_orth = np.cos(sum_angle_orth)

        J11 = -L1*sin_angle - L2*sin_sum_angle - L2ptr*sin_sum_angle_orth
        J12 = -L2*sin_sum_angle - L2ptr*sin_sum_angle_orth
        J21 =  L1*cos_angle + L2*cos_sum_angle + L2ptr*cos_sum_angle_orth;
        J22 =  L2*cos_sum_angle + L2ptr*cos_sum_angle_orth 

        x_vel = J11*sh_vel + J12*el_vel;
        y_vel = J21*sh_vel + J22*el_vel;

    if not (sh_acc == None or el_acc == None) and arm == 'right':
        x_acc = J11*sh_acc + J12*el_acc;
        y_acc = J21*sh_acc + J22*el_acc;
        
    if sh_vel == None:
        return (x_pos, y_pos)
    elif sh_acc == None:        
        return (x_pos, y_pos, x_vel, y_vel)
    else:
        return (x_pos, y_pos, x_vel, y_vel, x_acc, y_acc)

def pol2cart(theta, r):
    return r*np.cos(theta), r*np.sin(theta)

