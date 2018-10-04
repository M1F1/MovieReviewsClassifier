import os
import sys
from numpy import random as rnd
from numpy import genfromtxt
import time
import pandas as pd
import sys
sys.path.insert(0, 'C:\\Users\\Qbit\\Inzynierka')
from myTools import  moving_commands as mc
import glob
from random import randrange
import random
import subprocess
import random

mc.move_to_model_location()
cwd = os.getcwd()
nn = sys.argv[1]
values = []
model_dir = os.path.abspath(os.path.join(cwd, nn))
os.chdir(model_dir)
runs_dir = os.path.join(model_dir, 'runs')
hyper_search_dir = os.path.join(model_dir, 'hyperparameters_search')
list_of_models = glob.glob(runs_dir + '\\*')
time_stamp = str(int(time.time()))
for model in list_of_models:
	start_time = time.time()
	batch_size = model.split("_")[-1][2:]
	print("batch size", batch_size)
	subprocess.call(['python', 'eval_on_train.py', '--batch_size=' + str(batch_size), '--model_name=' + model], shell=True)
	elapsed_testing_time = time.time() - start_time
	accuracy = genfromtxt(os.path.join(runs_dir, model, 'checkpoints', 'accuracy_on_train.csv'))
	values.append([accuracy, model, elapsed_testing_time])

df = pd.DataFrame(values, columns=['accuracy', 'model_name', 'elapsed_testing_time'])
df.to_csv(os.path.join(hyper_search_dir, time_stamp +'eval_on_train'+ '.csv'), index=False)

print(df)
