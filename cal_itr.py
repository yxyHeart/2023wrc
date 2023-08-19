
import numpy as np
import math
from scipy.special import softmax
from decimal import Decimal
import sys
from loguru import logger

def cal_itr(T,N,P):
    ITR = (1 / T) * (math.log(N, 2) + (1 - P) * math.log((1 - P) / (N - 1), 2) + P * math.log(P, 2)) * 60
    print(ITR)

cal_itr(1.7, 40, 0.78)
