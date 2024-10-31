import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/install/socspioneer'
