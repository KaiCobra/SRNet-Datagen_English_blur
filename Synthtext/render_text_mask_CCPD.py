"""
Rendering text mask.
Change the original code to Python3 support and simplifed the code structure.
Original project: https://github.com/ankush-me/SynthText
Author: Ankush Gupta
Date: 2015
"""

import os
import cv2
import time
import random
import math
import numpy as np
import pygame, pygame.locals
from pygame import freetype
from math import pi
from . import data_cfg

def center2size(surf, size,text_bbox,char_bbox_list):
    # for i in char_bbox_list:
    #     lt, rt, rb, lb = i
    #     lt, rt, rb, lb = tuple(lt), tuple(rt), tuple(rb), tuple(lb)
    #     cv2.line(surf, lt, rt, 127, 2)
    #     cv2.line(surf, rt, rb, 127, 2)
    #     cv2.line(surf, rb, lb, 127, 2)
    #     cv2.line(surf, lb, lt, 127, 2)
    #     cv2.imwrite('/media/avlab/disk3/LP2022_temp_test/test_img/perspective2.png',surf)
    canvas = np.zeros(size).astype(np.uint8)
    size_h, size_w = size
    surf_h, surf_w = surf.shape[:2]
    padding_h = (size_h-surf_h)//2
    padding_w = (size_w-surf_w)//2
    text_bbox = [[point[0]+padding_w,point[1]+padding_h] for point in text_bbox]
    
    
    for idx,char_bbox_inf in enumerate(char_bbox_list):
        char_bbox_list[idx] = [[x+padding_w,y+padding_h] for [x,y] in char_bbox_inf]

    # new_char_bbox_list = bbox_move(char_bbox_list,(new_w,new_h))
    

    canvas[padding_h:padding_h+surf_h, padding_w:padding_w+surf_w] = surf
    # for i in char_bbox_list:
    #     lt, rt, rb, lb = i
    #     lt, rt, rb, lb = tuple(lt), tuple(rt), tuple(rb), tuple(lb)
    #     cv2.line(canvas, lt, rt, 127, 2)
    #     cv2.line(canvas, rt, rb, 127, 2)
    #     cv2.line(canvas, rb, lb, 127, 2)
    #     cv2.line(canvas, lb, lt, 127, 2)
    #     cv2.imwrite('/media/avlab/disk3/LP2022_temp_test/test_img/perspective3.png',canvas)

    # lt, rt, rb, lb = text_bbox
    # lt, rt, rb, lb = tuple(lt), tuple(rt), tuple(rb), tuple(lb)
    # cv2.line(canvas, lt, rt, 127, 2)
    # cv2.line(canvas, rt, rb, 127, 2)
    # cv2.line(canvas, rb, lb, 127, 2)
    # cv2.line(canvas, lb, lt, 127, 2)
    # cv2.imwrite('/media/avlab/disk3/LP2022_temp_test/test_img/perspective2_test.png',canvas)

    
    return canvas,text_bbox,char_bbox_list


def crop_safe(arr, rect, bbs=[], pad=0):
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2*pad
    v0 = [max(0,rect[0]), max(0,rect[1])]
    v1 = [min(arr.shape[0], rect[0]+rect[2]), min(arr.shape[1], rect[1]+rect[3])]
    
    
    arr = arr[v0[0]:v1[0], v0[1]:v1[1], ...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i,0] -= v0[0]
            bbs[i,1] -= v0[1]
        return arr, bbs
    else:
        return arr
def interval(ch, text_len):

    if text_len == 8:
    ##### 7 character
        interval_bf = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,  'F': 0, 'G': 0,
                    'H': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0,  'N': 0, 'P': 0, 
                    'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0,  'V': 0, 'W': 0, 
                    'X': 0, 'Y': 0, 'Z': 0, '0': 0, '1': 1, '2': 0, '3': 0, 
                    '4': 0, '5': 0, '6': 0, '7': 0, '8': 0,  '9': 0, '·': 0}
        interval_af = {'A': 28, 'B': 28, 'C': 28, 'D': 28, 'E': 28, 'F': 28, 'G': 28,
                    'H': 28, 'J': 28, 'K': 28, 'L': 28, 'M': 28, 'N': 28, 'P': 28, 
                    'Q': 28, 'R': 28, 'S': 28, 'T': 28, 'U': 28, 'V': 28, 'W': 28, 
                    'X': 28, 'Y': 28, 'Z': 28, '0': 28, '1': 28, '2': 28, '3': 28, 
                    '4': 28, '5': 28, '6': 28, '7': 28, '8': 28, '9': 28, '·': 10}
    elif text_len == 7:
    ##### 6 character
        interval_bf = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,  'F': 0, 'G': 0,
                    'H': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0,  'N': 0, 'P': 0, 
                    'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0,  'V': 0, 'W': 0, 
                    'X': 0, 'Y': 0, 'Z': 0, '0': 0, '1': 0, '2': 0, '3': 0, 
                    '4': 0, '5': 0, '6': 0, '7': 0, '8': 0,  '9': 0, '·': -3}
        interval_af = {'A': 30, 'B': 30, 'C': 30, 'D': 30, 'E': 30, 'F': 30, 'G': 30,
                    'H': 30, 'J': 30, 'K': 30, 'L': 30, 'M': 30, 'N': 30, 'P': 30, 
                    'Q': 30, 'R': 30, 'S': 30, 'T': 30, 'U': 30, 'V': 30, 'W': 30, 
                    'X': 30, 'Y': 30, 'Z': 30, '0': 30, '1': 30, '2': 30, '3': 30, 
                    '4': 30, '5': 30, '6': 30, '7': 30, '8': 30, '9': 30, '·': 10}

    elif text_len == 6:
    ##### 5 character
        interval_bf = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,  'F': 0, 'G': 0,
                    'H': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0,  'N': 0, 'P': 0, 
                    'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0,  'V': 0, 'W': 0, 
                    'X': 0, 'Y': 0, 'Z': 0, '0': 0, '1': 1, '2': 0, '3': 0, 
                    '4': 0, '5': 0, '6': 0, '7': 0, '8': 0,  '9': 0, '-': -10}
        interval_af = {'A': 33, 'B': 33, 'C': 33, 'D': 33, 'E': 33, 'F': 33, 'G': 33,
                    'H': 33, 'J': 33, 'K': 33, 'L': 33, 'M': 33, 'N': 33, 'P': 33, 
                    'Q': 33, 'R': 33, 'S': 33, 'T': 33, 'U': 33, 'V': 33, 'W': 33, 
                    'X': 33, 'Y': 33, 'Z': 33, '0': 33, '1': 33, '2': 33, '3': 33, 
                    '4': 33, '5': 33, '6': 33, '7': 33, '8': 33, '9': 33, '-': 23}

    elif text_len == 5:
    ##### 4 character
        interval_bf = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,  'F': 0, 'G': 0,
                    'H': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0,  'N': 0, 'P': 0, 
                    'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0,  'V': 0, 'W': 0, 
                    'X': 0, 'Y': 0, 'Z': 0, '0': 0, '1': 1, '2': 0, '3': 0, 
                    '4': 0, '5': 0, '6': 0, '7': 0, '8': 0,  '9': 0, '·': 0}
        interval_af = {'A': 28, 'B': 28, 'C': 28, 'D': 28, 'E': 28, 'F': 28, 'G': 28,
                    'H': 28, 'J': 28, 'K': 28, 'L': 28, 'M': 28, 'N': 28, 'P': 28, 
                    'Q': 28, 'R': 28, 'S': 28, 'T': 28, 'U': 28, 'V': 28, 'W': 28, 
                    'X': 28, 'Y': 28, 'Z': 28, '0': 29, '1': 28, '2': 28, '3': 28, 
                    '4': 28, '5': 28, '6': 28, '7': 28, '8': 28, '9': 28, '·': 10}

    

    return interval_bf[ch],interval_af[ch]

def render_normal(font, text):
        
    # get the number of lines

    lines = text.split('\n')

    lengths = [len(l) for l in lines]

    # font parameters:
    line_spacing = font.get_sized_height()
    # initialize the surface to proper size:
    line_bounds = font.get_rect(lines[np.argmax(lengths)])
    fsize = (round(2.0 * line_bounds.width), round(1.25 * line_spacing * len(lines)))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    bbs = []
    space = font.get_rect('O')
    x, y = 0, 0
    char_bbox_list = []
    for l in lines:
        x = 0 # carriage-return
        y += line_spacing # line-feed
        for ch in l: # render each character
            if ch.isspace(): # just shift
                x += space.width
            else:
                # render the character

                ch_bounds = font.render_to(surf, (x,y), ch)
                lt = [int(x),int(y - ch_bounds.y)]
                rt = [int(x+ch_bounds.width),int(y - ch_bounds.y)]
                rb = [int(x+ch_bounds.width),int(y - ch_bounds.y+ch_bounds.height)]
                lb = [int(x),int(y - ch_bounds.y+ch_bounds.height)]
                char_bbox_inf = [lt,rt,rb,lb]
                char_bbox_list.append(char_bbox_inf)

                ch_bounds.x = x + ch_bounds.x
                ch_bounds.y = y - ch_bounds.y
                
                x += ch_bounds.width
                bbs.append(np.array(ch_bounds))   
                bbs.append(np.array(ch_bounds))


    # get the union of characters for cropping:
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)

    # get the words:
    words = ' '.join(text.split())

    # crop the surface to fit the text:
    bbs = np.array(bbs)
    surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=100 )
    surf_arr = surf_arr.swapaxes(0,1)
    
    loc_ori = np.where(surf_arr > 127)
    text_min_y, text_min_x = np.min(loc_ori[0]), np.min(loc_ori[1])
    text_max_y, text_max_x = np.max(loc_ori[0]), np.max(loc_ori[1])
    
    padding_x = min(text_min_x,(surf_arr.shape[1]-text_max_x))
    padding_y = min(text_min_y,(surf_arr.shape[0]-text_max_y))
    
    text_min_y =text_min_y-padding_y
    text_min_x= text_min_x-padding_x
    
    new_surf_arr = surf_arr[text_min_y:text_max_y + padding_y, text_min_x:text_max_x+padding_x]
    surf_arr2 = np.array(new_surf_arr)

    # cv2.imwrite('/home/avlab/scenetext/SRNet-Datagen_English/test.png',surf_arr2)

    char_bbox_list = bbox_move(char_bbox_list,(-text_min_x,-text_min_y))
    

            
    # for i in char_bbox_list:
    #     lt, rt, rb, lb = i
    #     lt, rt, rb, lb = tuple(lt), tuple(rt), tuple(rb), tuple(lb)
    #     cv2.line(surf_arr2, lt, rt, 127, 2)
    #     cv2.line(surf_arr2, rt, rb, 127, 2)
    #     cv2.line(surf_arr2, rb, lb, 127, 2)
    #     cv2.line(surf_arr2, lb, lt, 127, 2)
    #     cv2.imwrite('/media/avlab/disk3/LP2022_temp_test/test_img/test2.png',surf_arr2)
    # a = 'b'

    # new_surf_arr = np.pad(new_surf_arr,200)
    return new_surf_arr, bbs, char_bbox_list

def bbox_move(char_bbox_list,move):
    move_x,move_y = move
    for idx,char_bbox_inf in enumerate(char_bbox_list):
        char_bbox_list[idx] = [[x+move_x,y+move_y] for [x,y] in char_bbox_inf]
    return char_bbox_list

def render_curved(font, text, curve_rate, curve_center = None):

    wl = len(text)

    # create the surface:
    lspace = font.get_sized_height() + 1
    lbound = font.get_rect(text)
    #fsize = (round(2.0*lbound.width), round(3*lspace))
    fsize = (round(3.0*lbound.width), round(5*lspace))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

    # baseline state
    if curve_center is None:
        curve_center = wl // 2
    curve_center = max(curve_center, 0)
    curve_center = min(curve_center, wl - 1)
    mid_idx = curve_center #wl//2
    curve = [curve_rate * (i - mid_idx) * (i - mid_idx) for i in range(wl)]
    curve[mid_idx] = -np.sum(curve) / max(wl-1, 1)
    rots  = [-int(math.degrees(math.atan(2 * curve_rate * (i-mid_idx)/(font.size/2)))) for i in range(wl)]

    bbs = []
    # place middle char
    rect = font.get_rect(text[mid_idx])
    rect.centerx = surf.get_rect().centerx
    rect.centery = surf.get_rect().centery + rect.height
    rect.centery +=  curve[mid_idx]
    ch_bounds = font.render_to(surf, rect, text[mid_idx], rotation = rots[mid_idx])
    ch_bounds.x = rect.x + ch_bounds.x
    ch_bounds.y = rect.y - ch_bounds.y
    mid_ch_bb = np.array(ch_bounds)

    # render chars to the left and right:
    last_rect = rect
    ch_idx = []
    for i in range(wl):
        #skip the middle character
        if i == mid_idx:
            bbs.append(mid_ch_bb)
            ch_idx.append(i)
            continue

        if i < mid_idx: #left-chars
            i = mid_idx-1-i
        elif i == mid_idx + 1: #right-chars begin
            last_rect = rect

        ch_idx.append(i)
        ch = text[i]

        newrect = font.get_rect(ch)
        newrect.y = last_rect.y
        if i > mid_idx:
            newrect.topleft = (last_rect.topright[0] + 2, newrect.topleft[1])
        else:
            newrect.topright = (last_rect.topleft[0] - 2, newrect.topleft[1])
        newrect.centery = max(newrect.height, min(fsize[1] - newrect.height, newrect.centery + curve[i]))
        try:
            bbrect = font.render_to(surf, newrect, ch, rotation = rots[i])
        except ValueError:
            bbrect = font.render_to(surf, newrect, ch)
        bbrect.x = newrect.x + bbrect.x
        bbrect.y = newrect.y - bbrect.y
        bbs.append(np.array(bbrect))
        last_rect = newrect

    # correct the bounding-box order:
    bbs_sequence_order = [None for i in ch_idx]
    for idx,i in enumerate(ch_idx):
        bbs_sequence_order[i] = bbs[idx]
    bbs = bbs_sequence_order

    # get the union of characters for cropping:
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)

    # crop the surface to fit the text:
    bbs = np.array(bbs)
    surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad = 5)
    surf_arr = surf_arr.swapaxes(0,1)
    return surf_arr, bbs

def center_warpPerspective(img, H, center, size):

    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype = np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    img2 = cv2.warpPerspective(img, M, size,cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)

    img3 = cv2.warpPerspective(img, H, size)                
    return img2

def center_pointsPerspective(points, H, center):

    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype = np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    return M.dot(points)


def get_M(img_h, img_w,focal, theta, phi, gamma, dx, dy, dz):
        w = img_w
        h = img_h
        f = focal
        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        # print(np.cos(phi),np.sin(phi))
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)
        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])
        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])
        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))

def deg_to_rad(deg):
    return deg * pi / 180.0

def rad_to_deg(rad):
    return rad * 180.0 / pi

def get_transform_martix(img_h,img_w,theta,phi,gamma,focal):
    dx = 0
    dy = 0
    # d = np.sqrt(img_h**2 + img_w**2)
    rtheta, rphi, rgamma = get_rad(theta, phi, gamma)
    # focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    # focal = 500
    dz = focal
    M = get_M(img_h, img_w,focal,rtheta, rphi, rgamma, dx, dy, dz)
    return M
    
# def check_angle(img,angle):
#     theta,phi,gamma,focal = angle
#     img_h, img_w = img.shape[:2]
    
#     M = get_transform_martix(img_h,img_w,theta,phi,gamma,focal)

#     img_h, img_w = img.shape[:2]
#     img_center = (img_w / 2, img_h / 2)
#     a = 0
#     while a == 0:
#         points = np.ones((3, 4), dtype = np.float32)
#         points[:2, 0] = np.array([0, 0], dtype = np.float32).T
#         points[:2, 1] = np.array([img_w, 0], dtype = np.float32).T
#         points[:2, 2] = np.array([img_w, img_h], dtype = np.float32).T
#         points[:2, 3] = np.array([0, img_h], dtype = np.float32).T

        
#         perspected_points = center_pointsPerspective(points, M, img_center)
        
#         perspected_points[0, :] /= perspected_points[2, :]
#         perspected_points[1, :] /= perspected_points[2, :]
        
        
#         canvas_w = int(2 * max(img_center[0], img_center[0] - np.min(perspected_points[0, :]), 
#                         np.max(perspected_points[0, :]) - img_center[0])) + 10
#         canvas_h = int(2 * max(img_center[1], img_center[1] - np.min(perspected_points[1, :]), 
#                         np.max(perspected_points[1, :]) - img_center[1])) + 10
#         return canvas_w,canvas_h

# def angle(img1,img2, angle): # w first
#     a = 0
#     while a == 0:
#         canvas1_w,canvas1_h = check_angle(img1,angle)
#         canvas2_w,canvas2_h = check_angle(img2,angle)
#         if canvas1_w<1000 and canvas1_h<1000:
#             if canvas2_w<1000 and canvas2_h<1000:
#                 return angle
#         theta = np.random.randint(-10, 10)
#         phi = np.random.randint(-10, 10)
#         gamma = np.random.randint(-45, 45)
#         angle = [theta,phi,gamma]

def bbox_transform(M,char_bbox_list):
    char_perspected_bbox = []
    for char_bbox in char_bbox_list:
        [lt,rt,rb,lb]= char_bbox
        points_char = np.ones((3, 4), dtype = np.float32)
        points_char[:2, 0] = np.array(lt, dtype = np.float32).T
        points_char[:2, 1] = np.array(rt, dtype = np.float32).T
        points_char[:2, 2] = np.array(rb, dtype = np.float32).T
        points_char[:2, 3] = np.array(lb, dtype = np.float32).T

        char_perspected_points = M.dot(points_char)

        char_perspected_points[0, :] /= char_perspected_points[2, :]
        char_perspected_points[1, :] /= char_perspected_points[2, :]
        temp_list = []
        for i in range(4):
            [x,y]= char_perspected_points[:2, i]
            temp_list.append([int(x),int(y)])
        char_perspected_bbox.append(temp_list)
        
    return char_perspected_bbox


    # create padded destination image

def warpPerspectivePadded(
        src, dst, M, char_bbox_list,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0):
    """Performs a perspective warp with padding.
    Parameters
    ----------
    src : array_like
        source image, to be warped.
    dst : array_like
        destination image, to be padded.
    M : array_like
        `3x3` perspective transformation matrix.
    Returns
    -------
    src_warped : ndarray
        padded and warped source image
    dst_padded : ndarray
        padded destination image, same size as src_warped
    Optional Parameters
    -------------------
    flags : int, optional
        combination of interpolation methods (`cv2.INTER_LINEAR` or
        `cv2.INTER_NEAREST`) and the optional flag `cv2.WARP_INVERSE_MAP`,
        that sets `M` as the inverse transformation (`dst` --> `src`).
    borderMode : int, optional
        pixel extrapolation method (`cv2.BORDER_CONSTANT` or
        `cv2.BORDER_REPLICATE`).
    borderValue : numeric, optional
        value used in case of a constant border; by default, it equals 0.
    See Also
    --------
    warpAffinePadded() : for `2x3` affine transformations
    cv2.warpPerspective(), cv2.warpAffine() : original OpenCV functions
    """

    assert M.shape == (3, 3), \
        'Perspective transformation shape should be (3, 3).\n' \
        + 'Use warpAffinePadded() for (2, 3) affine transformations.'

    M = M / M[2, 2]  # ensure a legal homography
    if flags in (cv2.WARP_INVERSE_MAP,
                 cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                 cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP):
        M = cv2.invert(M)[1]
        flags -= cv2.WARP_INVERSE_MAP

    # it is enough to find where the corners of the image go to find
    # the padding bounds; points in clockwise order from origin
    src_h, src_w = src.shape[:2]
    lin_homg_pts = np.array([
        [0, src_w, src_w, 0],
        [0, 0, src_h, src_h],
        [1, 1, 1, 1]])

    # transform points
    transf_lin_homg_pts = M.dot(lin_homg_pts)
    transf_lin_homg_pts /= transf_lin_homg_pts[2, :]

    

    # find min and max points
    min_x = np.floor(np.min(transf_lin_homg_pts[0])).astype(int)
    min_y = np.floor(np.min(transf_lin_homg_pts[1])).astype(int)
    max_x = np.ceil(np.max(transf_lin_homg_pts[0])).astype(int)
    max_y = np.ceil(np.max(transf_lin_homg_pts[1])).astype(int)

    # add translation to the transformation matrix to shift to positive values
    anchor_x, anchor_y = 0, 0
    transl_transf = np.eye(3, 3)
    if min_x < 0:
        anchor_x = -min_x
        transl_transf[0, 2] += anchor_x
    if min_y < 0:
        anchor_y = -min_y
        transl_transf[1, 2] += anchor_y
    shifted_transf = transl_transf.dot(M)
    shifted_transf /= shifted_transf[2, 2]

    # create padded destination image
    dst_h, dst_w = dst.shape[:2]

    pad_widths = [anchor_y, max(max_y, dst_h) - dst_h,
                  anchor_x, max(max_x, dst_w) - dst_w]

    dst_padded = cv2.copyMakeBorder(dst, *pad_widths,
                                    borderType=borderMode, value=borderValue)
    
    dst_pad_h, dst_pad_w = dst_padded.shape[:2]
    src_warped = cv2.warpPerspective(
        src, shifted_transf, (dst_pad_w, dst_pad_h),
        flags=flags, borderMode=borderMode, borderValue=borderValue)
    new_char_bbox_list = bbox_transform(shifted_transf,char_bbox_list)

    return dst_padded, src_warped, new_char_bbox_list

def perspective(img, angle, pad,char_bbox_list,text): # w first

    theta, phi, gamma,focal = angle
    img_h, img_w = img.shape[:2]
    
    M = get_transform_martix(img_h,img_w,theta,phi,gamma,focal) # pitch,yaw,roll
  
    # loc_ori = np.where(canvas > 127)
    img_h, img_w = img.shape[:2]
    img2,ressze_image,per_char_bbox_list = warpPerspectivePadded(img, img, M, char_bbox_list)   
    # cv2.imwrite('/media/avlab/disk3/LP2022_temp_test2/test.png',ressze_image)

    # loc = np.where(canvas > 127)
    loc = np.where(ressze_image > 127)
    miny, minx = np.min(loc[0]), np.min(loc[1])
    maxy, maxx = np.max(loc[0]), np.max(loc[1])
    text_w = maxx - minx + 1
    text_h = maxy - miny + 1

    resimg = np.zeros((text_h + pad[2] + pad[3], text_w + pad[0] + pad[1])).astype(np.uint8)
    resimg[pad[2]:pad[2]+text_h, pad[0]:pad[0]+text_w] = ressze_image[miny:maxy+1, minx:maxx+1]
    move_x = pad[0]-minx
    move_y = pad[2]-miny
    move_char_bbox_list = bbox_move(per_char_bbox_list,(move_x,move_y))

    text_lt = move_char_bbox_list[0][0]
    text_lb = move_char_bbox_list[0][-1]
    text_rt = move_char_bbox_list[-1][1]
    text_rb = move_char_bbox_list[-1][2]

    text_bbox = [text_lt,text_rt, text_rb, text_lb]


    return resimg, text_bbox, move_char_bbox_list
    # return resimg
def draw_char_bbox(arr,char_bbox_list):
    for i in char_bbox_list:
        lt, rt, rb, lb = i
        lt, rt, rb, lb = tuple(lt), tuple(rt), tuple(rb), tuple(lb)
        cv2.line(arr, lt, rt, 127, 2)
        cv2.line(arr, rt, rb, 127, 2)
        cv2.line(arr, rb, lb, 127, 2)
        cv2.line(arr, lb, lt, 127, 2)
        cv2.imwrite('/media/avlab/disk3/LP2022_temp_test/test_img/test.png',arr)

def render_text(font, text):
    

    return render_normal(font, text)
