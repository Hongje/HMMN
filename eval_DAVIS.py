from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

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


### My libs
from dataset import DAVIS_MO_Test
from model import HMMN


torch.set_grad_enabled(False) # Volatile

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-s", type=str, help="set", required=True)
    parser.add_argument("-y", type=int, help="year", required=True)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("-D", type=str, help="path to data",default='/local/DATA')
    return parser.parse_args()

args = get_arguments()

GPU = args.g
YEAR = args.y
SET = args.s
VIZ = args.viz
DATA_ROOT = args.D

# Model and version
MODEL = 'HMMN'
print(MODEL, ': Testing on DAVIS')

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

if VIZ:
    print('--- Produce mask overaid video outputs. Evaluation will run slow.')
    print('--- Require FFMPEG for encoding, Check folder ./viz')


palette = Image.open(DATA_ROOT + '/Annotations/480p/blackswan/00000.png').getpalette()

def Run_video(Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]

    for t in tqdm.tqdm(range(1, num_frames)):
        # memorize
        with torch.no_grad():
            prev_key4, prev_value4, prev_key3, prev_value3, prev_key2, prev_value2 = model(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects])) 

        if t-1 == 0: # only prev memory
            this_keys4, this_values4 = prev_key4, prev_value4
            this_keys3, this_values3 = prev_key3, prev_value3
            this_keys2, this_values2 = prev_key2, prev_value2
        else:
            this_keys4 = torch.cat([keys4, prev_key4], dim=2)
            this_values4 = torch.cat([values4, prev_value4], dim=2)
            this_keys3 = torch.cat([keys3, prev_key3], dim=2)
            this_values3 = torch.cat([values3, prev_value3], dim=2)
            this_keys2 = torch.cat([keys2, prev_key2], dim=2)
            this_values2 = torch.cat([values2, prev_value2], dim=2)
        
        # segment
        with torch.no_grad():
            logit = model(Fs[:,:,t], this_keys4, this_values4, this_keys3, this_values3, this_keys2, this_values2, torch.tensor([num_objects]))
        Es[:,:,t] = F.softmax(logit, dim=1)
        
        # update
        if t-1 in to_memorize:
            keys4, values4 = this_keys4, this_values4
        if t-1 == 0: # update only the first frame memory
            keys3, values3 = this_keys3, this_values3
            keys2, values2 = this_keys2, this_values2
        
    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es



Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = nn.DataParallel(HMMN())
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN

pth_path = 'HMMN_weights.pth'
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path, map_location='cpu'))

code_name = '{}_DAVIS_{}{}'.format(MODEL,YEAR,SET)
print('Start Testing:', code_name)


for seq, V in enumerate(Testloader):
    Fs, Ms, num_objects, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
    
    pred, Es = Run_video(Fs, Ms, num_frames, num_objects, Mem_every=5, Mem_number=None)
        
    # Save results for quantitative eval ######################
    test_path = os.path.join('./test', code_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for f in range(num_frames):
        img_E = Image.fromarray(pred[f])
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))

    if VIZ:
        from helpers import overlay_davis
        # visualize results #######################
        viz_path = os.path.join('./viz/', code_name, seq_name)
        if not os.path.exists(viz_path):
            os.makedirs(viz_path)

        for f in range(num_frames):
            pF = (Fs[0,:,f].permute(1,2,0).numpy() * 255.).astype(np.uint8)
            pE = pred[f]
            canvas = overlay_davis(pF, pE, palette)
            canvas = Image.fromarray(canvas)
            canvas.save(os.path.join(viz_path, 'f{}.jpg'.format(f)))

        vid_path = os.path.join('./viz/', code_name, '{}.mp4'.format(seq_name))
        frame_path = os.path.join('./viz/', code_name, seq_name, 'f%d.jpg')
        os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(frame_path, vid_path))



