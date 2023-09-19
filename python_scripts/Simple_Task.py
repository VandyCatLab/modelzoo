import numpy as np
#import js2py
from hubReps.py import 


#js2py.translate_file('example.js', 'example.py')

rep1, rep2 = return_two_reps()

def LBA_deterministic(d1, d2, k=0, b=1, t0=0):
    """
    Return response and response time for a 2 alternate decision task where
    each accumulator only differ in their drift rate
    """
    rt1 = ((b - k) / d1) + t0
    rt2 = ((b - k) / d2) + t0

    if rt1 < rt2:
        return 1, rt1
    else:
        return 2, rt2


rep1, rep2 = return_two_reps()

response, response_time = LBA_deterministic(d1, d2, k, b, t0)