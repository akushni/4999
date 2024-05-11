import numpy as np
import pandas as pd
import time as time

from allFunctions_SimAndProc import *

def convert_frac(string):
    temp=string.split('/')
    return int(temp[0])/int(temp[1])


t1 = time.time()
df = pd.read_csv('/Users/abigailkushnir/Downloads/frequencies.tab',sep='\t')
t2 = time.time()
print(f'Data read in {t2-t1} seconds')

