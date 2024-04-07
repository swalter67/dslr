import sys
import pandas as pd
import numpy as np

def open():
    if len(sys.argv) > 2 or len(sys.argv) < 2:
        print("should have just one csv for arg")
        exit()

    arg = sys.argv[1]    
    try:
        data = pd.read_csv(arg)
    except:
        print("no valid csv")
        exit()
    return data

def count(v):
    ct = 0
    try:
        for i in v:
            if not np.isnan(i):
                ct += 1
    except:
        return len(v)
    return ct

def mean(v):
    m = 0
    ct = 0
    for i in v:
        if not np.isnan(i):
            m += i
            ct += 1
    return m / ct 

           