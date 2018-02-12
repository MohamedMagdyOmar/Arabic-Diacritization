import os
from tensorflow.python.client import device_lib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

local_device_protos = device_lib.list_local_devices()
#print (x.name for x in local_device_protos if x.device_type == 'GPU')

print (device_lib.list_local_devices())