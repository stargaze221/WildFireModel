import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random

#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

from torch.autograd.functional import jacobian

data = torch.rand(1, 3, 20, 20) 


transition_layer = nn.Conv2d(3, 3, 3, stride=1, padding=1).to(DEVICE)


def temp_function(data):
    output = transition_layer(data)
    return output

temp = jacobian(temp_function, data)
print('input_size:', data.size())
print('jacobian_size:', temp.size())