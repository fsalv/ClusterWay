# Copyright 2022 Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import cv2

from utils.geometry import line_polar_to_cart


def visualize_mask(mask=None,points=None,wp=None,wp_class=None,centers=None,borders=None,path=None,rad=6,dim=(7,7),axis=True):
    """
        Visualize a binary mask (0:rows, 1:background) and possibly starting/ending points, wp, centers and borders
    """
    if mask is None:
        img = np.ones((800,800,3)).astype("float32")
    elif not(len(mask.shape)==3 and mask.shape[-1]==3):
        img = np.tile(mask[...,None],3).astype("float32")
    else:
        img=mask.copy()
    if borders is not None:
        img = plot_border(img,borders,show_img=False)
    if points is not None:
        for p in points:
            cv2.circle(img,(p[0][0],p[0][1]),rad,color=(1,0,0),thickness=-1)
            cv2.circle(img,(p[1][0],p[1][1]),rad,color=(0,0.5,1),thickness=-1)
    if wp is not None:
        for i,p in enumerate(wp):
            if wp_class is not None:
                if wp_class[i]:
                    color = (0,0,1)
                else:
                    color = (1,0,0)
            else:
                color = (0,204/255,0)
            cv2.circle(img,(p[0],p[1]),rad,color=color,thickness=-1)
    if centers is not None:
        for p in centers:
            cv2.circle(img,(int(round(p[0])),int(round(p[1]))),rad,color=(1,0.8,0),thickness=-1)   
    show(img,dim=dim,path=path,axis=axis,markersize=rad)
    


def visualize_image_with_mask(img,mask=None,points=None,wp=None,wp_class=None,path=None,rad=6,dim=(7,7),axis=True):
    """
        Visualize the binary mask (0:rows, 1:background) together with the original image
    """
    img2 = img.copy()
    if mask is not None:
        img2[np.bitwise_not(mask.astype("bool"))] = 1   
    if points is not None:
        for p in points:
            cv2.circle(img2,(p[0][0],p[0][1]),rad,color=(1,0,0),thickness=-1)
            cv2.circle(img2,(p[1][0],p[1][1]),rad,color=(0,0,1),thickness=-1)
    if wp is not None:
        for i,p in enumerate(wp):
            if wp_class is not None:
                if wp_class[i]:
                    color = (0,0,1)
                else:
                    color = (1,0,0)
            else:
                color = (0,204/255,0)
            cv2.circle(img2,(p[0],p[1]),rad,color=color,thickness=-1)        
    show(img2,path=path,dim=dim,axis=axis,markersize=rad)


    
def visualize_points(points,centers,H=800,W=800,dim=(7,7)):
    """
        Visualize centers and starting/ending points used to generate the mask
    """
    fig = np.ones((H,W,3),"float")

    for c in centers:
        c = (int(round(c[0])),int(round(c[1])))
        cv2.circle(fig,(c[0],c[1]),5,color=(0,1,0),thickness=-1)

    for p1,p2 in points:
        cv2.circle(fig,(p1[0],p1[1]),5,color=(1,0,0),thickness=-1)
        cv2.circle(fig,(p2[0],p2[1]),5,color=(0,0,1),thickness=-1)
    show(fig,dim)
                     


def draw_line(img,x,y,radius=3,color=(0,0,0)):
    """
        Draw a line in xy coordinates
    """
    img2 = img.copy()
    
    x,y = np.round(x).astype("int"),np.round(y).astype("int")
    for i,j in zip(x,y):
        cv2.circle(img2,(i,j),radius,color,thickness=-1)
    return img2


                     
def plot_border(img,lines,show_img=True):
    """
        Add borders to image: lines should be a list of (alpha,point)
    """
    img2 = img.copy()

    for l in lines:
        r = np.arange(1000)
        x,y = line_polar_to_cart(r,l[0],l[1])
        img2 = draw_line(img2,x,y)

    if show_img:
        show(img2)
    return img2

                     
    
def show(img,path=None,dim=(7,7),markersize=3,axis=True):
    """
        Show img with dim
    """
    plt.figure(figsize=dim)
    plt.imshow(img)
    if path is not None:
        plt.plot(path[:,0],path[:,1],'--r.',markersize=markersize)
    if not axis:
        plt.gca().axis("off")
    plt.show()
