import sys
sys.path.append('model')
sys.path.append('utils')

from DPR.utils.utils_SH import *

# other modules
import os
import numpy as np

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2

def get_shading(model, impath):

    if isinstance(impath, str):
        img = cv2.imread(impath)
        img = cv2.resize(img, (512, 512))
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        img = impath
        Lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL).cuda())
    
    sh = np.loadtxt('DPR/example_light.txt')
    sh = sh[0:9]
    sh = sh * 0.7
   
    sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    outputImg, outputSH  = model(inputL, sh, 0)

    return outputSH


if __name__ == '__main__':
    
    # load model
    from DPR.model.defineHourglass_512_gray_skip import *
    my_network = HourglassNet()
    my_network.load_state_dict(torch.load('DPR/trained_model/trained_model_03.t7'))
    my_network.cuda()
    my_network.train(False)

    rst = get_shading(my_network, 'test_dataset/children/00007.png')
    print(rst)
    
