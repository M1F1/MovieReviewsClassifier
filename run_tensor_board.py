import os
import sys

nn = sys.argv[1]
print(os.system('tensorboard --logdir=' +"C:\\Users\\Qbit\\Inzynierka\\Models\\" +str(nn)))
