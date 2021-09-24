from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
 
# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

from helpers import *
 
print('Hierarchical Memory Matching Network: initialized.')
 
class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float() # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024
        return r4, r3, r2, c1, f
 
class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.convMemory = nn.Conv2d(inplanes//2, planes, kernel_size=(3,3), padding=(1,1), stride=1, bias=False)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm, mem):
        s = self.convFS(f) + self.convMemory(mem)
        s = self.ResFS(s)
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2, mem3, mem2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4, mem3) # out: 1/8, 256
        m2 = self.RF2(r2, m3, mem2) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p



class Memory(nn.Module):
    def __init__(self, gaussian_kernel, gaussian_kernel_flow_window):
        super(Memory, self).__init__()
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_kernel_flow_window = gaussian_kernel_flow_window
        if self.gaussian_kernel != -1:
            self.feature_H = -1
            self.feature_W = -1
            if self.gaussian_kernel_flow_window != -1:
                self.H_flow = -1
                self.W_flow = -1
                self.T_flow = 1e+7
                self.B_flow = -1
 
    def apply_gaussian_kernel(self, corr, h, w, sigma_factor=1.):
        b, hwt, hw = corr.size()

        idx = corr.max(dim=2)[1] # b x h2 x w2
        idx_y = (idx // w).view(b, hwt, 1, 1).float()
        idx_x = (idx % w).view(b, hwt, 1, 1).float()
        
        if h != self.feature_H:
            self.feature_H = h
            y_tmp = np.linspace(0,h-1,h)
            self.y = ToCuda(torch.FloatTensor(y_tmp))
        y = self.y.view(1,1,h,1).expand(b, hwt, h, 1)

        if w != self.feature_W:
            self.feature_W = w
            x_tmp = np.linspace(0,w-1,w)
            self.x = ToCuda(torch.FloatTensor(x_tmp))
        x = self.x.view(1,1,1,w).expand(b, hwt, 1, w)
                
        gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * (self.gaussian_kernel*sigma_factor)**2))
        gauss_kernel = gauss_kernel.view(b, hwt, hw)

        return gauss_kernel, idx
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW

        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        
        if self.gaussian_kernel != -1:
            if self.gaussian_kernel_flow_window != -1:
                p_tmp = p[:,int(-H*W):].clone()
                if (self.B_flow != B) or (self.T_flow != T) or (self.H_flow != H) or (self.W_flow != W):
                    hide_non_local_qk_map_tmp = torch.ones(B,1,H,W,H,W).bool()
                    window_size_half = (self.gaussian_kernel_flow_window-1) // 2
                    for h_idx1 in range(H):
                        for w_idx1 in range(W):
                            h_left = max(h_idx1-window_size_half, 0)
                            h_right = h_idx1+window_size_half+1
                            w_left = max(w_idx1-window_size_half, 0)
                            w_right = w_idx1+window_size_half+1
                            hide_non_local_qk_map_tmp[:,0,h_idx1,w_idx1,h_left:h_right,w_left:w_right] = False
                    hide_non_local_qk_map_tmp = hide_non_local_qk_map_tmp.view(B,H*W,H*W)
                    self.hide_non_local_qk_map_flow = ToCuda(hide_non_local_qk_map_tmp)
                if (self.B_flow != B) or (self.T_flow > T) or (T==1) or (self.H_flow != H) or (self.W_flow != W):
                    self.max_idx_stacked = None
                p_tmp.masked_fill_(self.hide_non_local_qk_map_flow, float('-inf'))
                gauss_kernel_map, max_idx = self.apply_gaussian_kernel(p_tmp, h=H, w=W)
                if self.max_idx_stacked is None:
                    self.max_idx_stacked = max_idx
                else:
                    if self.T_flow == T:
                        self.max_idx_stacked = self.max_idx_stacked[:,:int(-H*W)]
                    self.max_idx_stacked = torch.gather(max_idx, dim=1, index=self.max_idx_stacked)
                    for t_ in range(1, T):
                        gauss_kernel_map_tmp, _ = self.apply_gaussian_kernel(p_tmp, h=H, w=W, sigma_factor=(t_*0.5)+1)
                        gauss_kernel_map_tmp = torch.gather(gauss_kernel_map_tmp, dim=1, index=self.max_idx_stacked[:,int((T-t_-1)*H*W):int((T-t_)*H*W)].unsqueeze(-1).expand(-1,-1,int(H*W)))
                        gauss_kernel_map = torch.cat((gauss_kernel_map_tmp, gauss_kernel_map), dim=1)
                    self.max_idx_stacked = torch.cat((self.max_idx_stacked, max_idx), dim=1)
                self.T_flow = T
                self.H_flow = H
                self.W_flow = W
                self.B_flow = B
            else:
                gauss_kernel_map, _ = self.apply_gaussian_kernel(p, h=H, w=W)
                
        p = F.softmax(p, dim=1) # b, THW, HW

        if self.gaussian_kernel != -1:
            p.mul_(gauss_kernel_map)
            p.div_(p.sum(dim=1, keepdim=True))

        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p


class Memory_topk(nn.Module):
    def __init__(self, topk_guided_num):
        super(Memory_topk, self).__init__()
        self.topk_guided_num = topk_guided_num
 
    def forward(self, m_in, m_out, q_in, qk_ref, qk_ref_topk_indices=None, qk_ref_topk_val=None, mem_dropout=None):  # m_in: o,c,t,h,w
        B_ori, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        _, THW_ref, HW_ref = qk_ref.size()
        resolution_ref = int(math.sqrt((H*W) // HW_ref))
        H_ref = H // resolution_ref
        W_ref = W // resolution_ref

        size = resolution_ref

        if qk_ref_topk_indices is None:
            qk_ref_topk_val, qk_ref_topk_indices = torch.topk(qk_ref.transpose(1,2), k=self.topk_guided_num, dim=2, sorted=True)
            topk_guided_num = self.topk_guided_num
        else:
            topk_guided_num = qk_ref_topk_indices.shape[2]

        B = B_ori
        qk_ref_selected = qk_ref
        qk_ref_topk_indices_selected = qk_ref_topk_indices
        m_in_selected = m_in
        m_out_selected = m_out
        q_in_selected = q_in

        ref = torch.zeros_like(qk_ref_selected.transpose(1,2))
        ref.scatter_(2, qk_ref_topk_indices_selected, 1.)
        ref = ref.view(B, H_ref, W_ref, T, H_ref, W_ref)

        idx_all = torch.nonzero(ref)
        idx = idx_all[:, 0], idx_all[:, 1], idx_all[:, 2], idx_all[:, 3], idx_all[:, 4], idx_all[:, 5]
        m_in_selected = m_in_selected.view(B,D_e,T,H_ref,size,W_ref,size).permute(0,2,3,5,4,6,1)[idx[0], idx[3], idx[4], idx[5]] # B*H/2*W/2*k, 2, 2, Cin
        m_in_selected = m_in_selected.reshape(B, H_ref, W_ref, topk_guided_num*size*size, D_e) # B, H/2, W/2, k*size*size, Cin
        q_in_selected = q_in_selected.view(B,D_e,H_ref,size,W_ref,size) # B, Cin, H/2, 2, W/2, 2
        q_in_selected = q_in_selected.permute(0,2,4,1,3,5) # B, H/2, W/2, Cin, 2, 2
        m_out_selected = m_out_selected.view(B,D_o,T,H_ref,size,W_ref,size).permute(0,2,3,5,4,6,1)[idx[0], idx[3], idx[4], idx[5]] # B*H/2*W/2*k, 2, 2, Cout
        m_out_selected = m_out_selected.reshape(B, H_ref, W_ref, topk_guided_num*size*size, D_o)

        p = torch.einsum('bhwnc,bhwcij->bhwijn', m_in_selected, q_in_selected)
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=-1)

        mem_out = torch.einsum('bhwnc,bhwijn->bchiwj', m_out_selected, p)
        mem_out = mem_out.reshape(B, D_o, H, W)

        mem_out_pad = mem_out

        return mem_out_pad, qk_ref_topk_indices[:,:,:max(topk_guided_num//4,1)], qk_ref_topk_val[:,:,:max(topk_guided_num//4,1)]


class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim, only_key=False):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.only_key = only_key
        if not self.only_key:
            self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x):
        k = self.Key(x)
        v = self.Value(x) if not self.only_key else None
        return k, v




class HMMN(nn.Module):
    def __init__(self):
        super(HMMN, self).__init__()
        self.Encoder_M = Encoder_M() 
        self.Encoder_Q = Encoder_Q() 

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_M_r3 = KeyValue(512, keydim=128, valdim=256)
        self.KV_Q_r3 = KeyValue(512, keydim=128, valdim=-1, only_key=True)
        self.KV_M_r2 = KeyValue(256, keydim=64, valdim=128)
        self.KV_Q_r2 = KeyValue(256, keydim=64, valdim=-1, only_key=True)

        self.Memory = Memory(gaussian_kernel=3, gaussian_kernel_flow_window=7)
        self.Memory_topk3 = Memory_topk(topk_guided_num=32)
        self.Memory_topk2 = Memory_topk(topk_guided_num=32//4)

        self.Decoder = Decoder(256)
 
    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            # pad_mem = ToCuda(torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            # pad_mem[0,1:num_objects+1,:,0] = mem
            pad_mem = mem.unsqueeze(2)
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, frame, masks, num_objects): 
        # memorize a frame 
        num_objects = num_objects[0].item()
        _, K, H, W = masks.shape # B = 1

        (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

        # make batch arg list
        B_list = {'f':[], 'm':[], 'o':[]}
        for o in range(1, num_objects+1): # 1 - no
            B_list['f'].append(frame)
            B_list['m'].append(masks[:,o])
            B_list['o'].append( (torch.sum(masks[:,1:o], dim=1) + \
                torch.sum(masks[:,o+1:num_objects+1], dim=1)).clamp(0,1) )

        # make Batch
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)

        r4, r3, r2, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'])
        k4, v4 = self.KV_M_r4(r4) # num_objects, 128 and 512, H/16, W/16
        k3, v3 = self.KV_M_r3(r3)
        k2, v2 = self.KV_M_r2(r2)
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects, K=K)
        k3, v3 = self.Pad_memory([k3, v3], num_objects=num_objects, K=K)
        k2, v2 = self.Pad_memory([k2, v2], num_objects=num_objects, K=K)
        return k4, v4, k3, v3, k2, v2

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = ToCuda(torch.zeros(1, num_objects+1, H, W)) 
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        em[0,1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit

    def segment(self, frame, keys4, values4, keys3, values3, keys2, values2, num_objects): 
        num_objects = num_objects[0].item()
        K, keydim, T, H, W = keys4.shape # B = 1
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        k3, _ = self.KV_Q_r3(r3)
        k2, _ = self.KV_Q_r2(r2)
        
        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
        k3e = k3.expand(num_objects,-1,-1,-1)
        k2e = k2.expand(num_objects,-1,-1,-1)
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)
        
        # memory select kv:(1, K, C, T, H, W)
        # m4, pm4 = self.Memory(keys4[0,1:num_objects+1], values4[0,1:num_objects+1], k4e, v4e)
        m4, pm4 = self.Memory(keys4, values4, k4e, v4e)
        B, THW_ref, HW_ref = pm4.size()
        if THW_ref > (HW_ref):
            pm4_for_topk = torch.cat((pm4[:,:HW_ref], pm4[:,-HW_ref:]), dim=1) # First and Prev
        else:
            pm4_for_topk = pm4

        # m3, next_topk_indices, next_topk_val = self.Memory_topk3(keys3[0,1:num_objects+1], values3[0,1:num_objects+1], k3e, pm4_for_topk)
        # m2, _, _ = self.Memory_topk2(keys2[0,1:num_objects+1], values2[0,1:num_objects+1], k2e, pm4_for_topk, next_topk_indices, next_topk_val)
        m3, next_topk_indices, next_topk_val = self.Memory_topk3(keys3, values3, k3e, pm4_for_topk)
        m2, _, _ = self.Memory_topk2(keys2, values2, k2e, pm4_for_topk, next_topk_indices, next_topk_val)

        logits = self.Decoder(m4, r3e, r2e, m3, m2)
        ps = F.softmax(logits, dim=1)[:,1] # no, h, w  
        #ps = indipendant possibility to belong to each object
        
        logit = self.Soft_aggregation(ps, K) # 1, K, H, W

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        return logit    

    def forward(self, *args, **kwargs):
        if args[1].dim() > 4: # keys
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)


