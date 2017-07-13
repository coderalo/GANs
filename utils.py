import numpy as np
import random
import argparse
import sys
import os
import time
import datetime
import json
from termcolor import cprint

def print_time_info(string):
    T = time.gmtime()
    Y, M, D = T.tm_year, T.tm_mon, T.tm_mday
    h, m, s = T.tm_hour, T.tm_min, T.tm_sec
    print("[{}-{:0>2}-{:0>2} {:0>2}:{:0>2}:{:0>2}] {}".format(Y, M, D, h, m, s, string))


