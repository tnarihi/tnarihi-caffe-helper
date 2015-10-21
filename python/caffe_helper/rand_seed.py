# rand_seed.py
import os

envseed = os.environ.get('TNARIHI_CAFFE_HELPER_SEED', None)
if envseed is not None:
    if envseed == 'rand':
        import time
        seed = int(time.time())
        del time
    else:
        seed = int(envseed)
else:
	seed = None
