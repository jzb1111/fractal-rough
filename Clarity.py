# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:14:12 2021

@author: asus
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from read_data import read_wave_data

def calc_clarity(audio,level,multi=1):
    calc_level_lis=[]
    high=len(audio)
    min_=min(audio)
    max_=max(audio)
    audio_size=[(int(high))*multi,len(audio)*multi]
    audio_im=np.zeros(audio_size)
    for i in range(len(audio)):
        now_point=int((audio[i]+abs(min_))/(max_-min_)*high*multi)
        if now_point>len(audio)-1:
            now_point=len(audio)-1
        audio_im[now_point][i*multi]=1
        if i!=len(audio)-1:
            next_point=int((audio[i+1]+abs(min_))/(max_-min_)*high*multi)
            if next_point>len(audio)-1:
                next_point=len(audio)-1
        else:
            next_point=now_point
        for j in range(multi):
            point=int(j/multi*(next_point-now_point)+now_point)
            audio_im[point][i*multi+j]=1
    #plt.imshow(audio_im)
    div_lis=[]
    for i in range(level):
        div_lis.append(1/(i+1))
    #print(div_lis)
    count_lis=[]
    for i in range(level):
        factor=int(div_lis[i]*len(audio))
        size=[factor,factor]
        audio_tmp=resize_audio(audio_im,size)
        #plt.figure(i)
        #plt.imshow(audio_tmp)
        count_lis.append(count_one(audio_tmp))
    power_lis=[]
    for i in range(len(count_lis)):
        #power=np.log(count_lis[i])/np.log(count_lis[0])
        if i!=0:
            power=np.log(count_lis[0]/count_lis[i])/np.log(1/div_lis[i])
            power_lis.append(power)
    #print(power_lis)
    return power_lis 

def resize_audio(audio_image,size):
    tmp=np.zeros(size)
    one_lis=[]
    for i in range(len(audio_image)):
        for j in range(len(audio_image[i])):
            if audio_image[j][i]==1:
                one_lis.append(j)
    #one_lis=connect_point(one_lis)
    #print(one_lis)
    for i in range(size[1]):
        ind=int(i/size[1]*len(audio_image))
        
        point=int(one_lis[ind]*size[0]/len(audio_image[0]))
        if point>size[0]-1:
            point=size[0]-1
        #print(point,ind)
        tmp[point][i]=1
    tmp_one_lis=[]
    for i in range(len(tmp)):
        for j in range(len(tmp[i])):
            if tmp[j][i]==1:
                tmp_one_lis.append(j)
    res_lis=connect_point(tmp_one_lis)
    res=np.zeros(size)
    for i in range(len(res_lis)):
        res[res_lis[i][1]][res_lis[i][0]]=1
    return res

def count_one(im):
    count=0
    for i in range(len(im)):
        for j in range(len(im[i])):
            if im[i][j]==1:
                count+=1
    return count

def count_dist(im):
    dist=0
    for i in range(len(im)):
        for j in range(len(im[i])):
            if im[j][i]==1:
                now_point=j
                if i!=len(im)-1:
                    for k in range(len(im)):
                        if im[k][i]==1:
                            next_point=k
                            dist+=((next_point-now_point)**2+1)**0.5
    return dist

def connect_point(lis):
    res=[]
    for i in range(len(lis)):
        now_point=[i,lis[i]]
        if i!=len(lis)-1:
            next_point=[i+1,lis[i+1]]
            plus=[]
            point_tmp=now_point
            while point_tmp!=next_point:
                plus.append(point_tmp)
                select_lis=[[point_tmp[0]-1,point_tmp[1]-1],[point_tmp[0],point_tmp[1]-1],[point_tmp[0]+1,point_tmp[1]-1],[point_tmp[0]-1,point_tmp[1]],[point_tmp[0]-1,point_tmp[1]+1],[point_tmp[0],point_tmp[1]+1],[point_tmp[0]+1,point_tmp[1]+1],[point_tmp[0]+1,point_tmp[1]]]
                dist_lis=[(next_point[0]-i[0])**2+(next_point[1]-i[1])**2 for i in select_lis]
                min_point=select_lis[dist_lis.index(min(dist_lis))]
                #print(now_point,min_point,next_point)
                point_tmp=min_point
            res+=plus
    return res

'''audio1,f1=read_wave_data('./binaural_dataset/trainset/subject1/mono.wav')
t=15
l=900
sig=audio1[t*l:(t+1)*l]
c_lis=calc_clarity(sig, level=10,multi=1)
print(np.mean(c_lis))'''

sig1=[np.sin(i/100) for i in range(628)]
plt.figure(1)
plt.plot(sig1)
c_lis1=calc_clarity(sig1, level=10,multi=1)
print(np.mean(c_lis1))

sig2=[np.sin(i/100)/3+np.sin(i/50)/3+np.sin(i/10)/3 for i in range(628)]
plt.figure(2)
plt.plot(sig2)
c_lis2=calc_clarity(sig2, level=10,multi=1)
print(np.mean(c_lis2))

sig3=[np.sin(i/100)/4+np.sin(i/50)/4+np.sin(i/10)/4+np.sin(i/5)/4 for i in range(628)]
plt.figure(3)
plt.plot(sig3)
c_lis3=calc_clarity(sig3, level=10,multi=1)
print(np.mean(c_lis3))