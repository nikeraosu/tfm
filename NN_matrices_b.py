#!/usr/bin/env python3

#After merging all the matrices it splits it into train&validation/test matrices


import os
import pandas as pd
import json
import optparse
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split

def addOptions(parser):
   parser.add_option("--NNconfigFile", default="",
             help="Neural Network config json file for the different parameters")

parser = optparse.OptionParser()
addOptions(parser)

(options, args) = parser.parse_args()

if not options.NNconfigFile:
   print >> sys.stderr, "No configuration file specified\n"
   sys.exit(1)

#with open('config.json', 'r') as cfg_file:
with open(options.NNconfigFile, 'r') as cfg_file:
    cfg_data = json.load(cfg_file)

orig_folder = cfg_data['orig_folder']
dest_folder = cfg_data['dest_folder']
der_folder = cfg_data['der_folder']
derivative = cfg_data['derivative']
scores_file = cfg_data['scores_file']
score_to_use = cfg_data['score_to_use']
scores_threshold = cfg_data['scores_threshold'] #we set 25% as our default threshold and get the score for it


input_files=[]
input_files_dates=[]
in_files = os.listdir(orig_folder)
for input_file in in_files:
    if input_file.endswith('.csv'):
        input_files.append(input_file)
print('len input files: {}\n.'.format(len(input_files)))
#We remove the vulcan activity &/or weird days:

scores = pd.read_csv(scores_file)

scores_days = scores['date']
scores_thr = scores[score_to_use]
#Now we get... 
scores_thr_index = scores_thr.index.values
days_thr = scores_thr_index[scores_thr > scores_threshold]

#And then we end up with a list of days (int)

bad_days = scores_days[days_thr]
days_list = [] #in str
for i in bad_days:
    days_list.append(str(i))
print('Bad days to be removed {}'.format(days_list))
#Finally we remove our bad days and end up with "kind of good" days (at least days with the same behaviour, we think)
for d in days_list:
    for f in input_files:
        if d in f:
            input_files.remove(f)


lenfiles = len(input_files)
print('lenfiles: {}\n'.format(lenfiles))
input_der_files = []
if derivative == True:
    print('derivative TRUE\n')
    for i in input_files:
        input_der_files.append(der_folder + i.replace('.csv','_der.csv'))
else:
    print('derivative FALSE\n')
tv = dest_folder + '/train_validation'
train = dest_folder + '/train_validation/train'
validation = dest_folder + '/train_validation/validation'
test = dest_folder + '/test'

if not os.path.exists(train):
    os.makedirs(train)
if not os.path.exists(tv):
    os.makedirs(tv)
if not os.path.exists(validation):
    os.makedirs(validation)
if not os.path.exists(test):
    os.makedirs(test)

X = pd.DataFrame()
Y = pd.DataFrame()

#Randomize days:


arr_days = np.arange(lenfiles)
ran_seed = 42 #our seed to randomize data
np.random.seed(ran_seed)
np.random.shuffle(arr_days)

lenfiles_test = int(round(lenfiles * 0.15,0))
arr_rand_test = arr_days[0:lenfiles_test]

lenfiles_train = lenfiles - lenfiles_test
arr_rand_train = arr_days[lenfiles_test:]

day0 = pd.read_csv(orig_folder + '/' + input_files[0])
lenday = len(day0)

input_files_test = []
input_files_train = []
dates_file = pd.read_csv('../experimentos/paper_train_validation_base/test_days.csv')
#print(dates_file)
input_date_test = [date[4:12] for date in np.array(dates_file['test_days'].values)]
#print(input_date_test)
#for i in arr_rand_test:
#    input_files_test.append(input_files[i])
input_files_test = []
for i in input_files:
    #print(i[4:12])
    if i[4:12] in input_date_test:
        input_files_test.append(i)
##input_files_test = [data_file for input_files if data_file[] in input_date_test ]
#for i in arr_rand_train:
  #  input_files_train.append(input_files[i])
for i in input_files:
    if i[4:12] not in input_date_test:
        input_files_train.append(i)
print('test {}'.format(input_files_test))
print('train {}'.format(input_files_train))
#Normalization for azimut and elevation here. Should've been done before, right... TO DO
#Initialization first...
az_max = -999
az_min = 999
ele_max = -999
ele_min = 999
#Then we calculate the max and mins and update it. We normalize after the matrices are built, in the last part of the code.

print('Appending {} test files\n'.format(lenfiles_test))
start_time = time.time()
for f in input_files_test:
    df = pd.read_csv(orig_folder + '/' + f)
    x = df[df.columns[0:-1]].round(5)
    #Here's the calculation for the normalization...
    if x.azimut.max() > az_max:
        az_max = x.azimut.max()
    if x.azimut.min() < az_min:
        az_min = x.azimut.min()
    if x.elevation.max() > ele_max:
        ele_max = x.elevation.max()
    if x.elevation.min() < ele_min:
        ele_min = x.elevation.min()
    #...And here it ends.

    y = df[df.columns[-1]].round(5)
    if derivative == True:
        f_der = f.replace('.csv','_der.csv')
        d = pd.read_csv(der_folder + '/' + f_der).round(5)
        x = pd.concat([x,d],axis=1)

    X = pd.concat([X,x],ignore_index=True)
    Y = pd.concat([Y,y],ignore_index=True)

X.to_csv(test+'/X_test.csv',header=True,index=False)
Y.to_csv(test+'/Y_test.csv',header=True,index=False)

end_time = time.time()
print('{} files appended in {} seconds. Splitting matrices.\n'.format(lenfiles_test,end_time-start_time))

X = pd.DataFrame()
Y = pd.DataFrame()
print('Appending {} train and validation files\n'.format(lenfiles_train))
start_time = time.time()
for f in input_files_train:
    df = pd.read_csv(orig_folder + '/' + f)
    x = df[df.columns[0:-1]].round(5)
    #Here's the calculation for the normalization...
    if x.azimut.max() > az_max:
        az_max = x.azimut.max()
    if x.azimut.min() < az_min:
        az_min = x.azimut.min()
    if x.elevation.max() > ele_max:
        ele_max = x.elevation.max()
    if x.elevation.min() < ele_min:
        ele_min = x.elevation.min()
    #...And here it ends.
 
    y = df[df.columns[-1]].round(5)
    if derivative == True:
        f_der = f.replace('.csv','_der.csv')
        d = pd.read_csv(der_folder + '/' + f_der).round(5)
        x = pd.concat([x,d],axis=1)

    X = pd.concat([X,x],ignore_index=True)
    Y = pd.concat([Y,y],ignore_index=True)


X.to_csv(tv+'/X_tr_val.csv',header=True,index=False)
Y.to_csv(tv+'/Y_tr_val.csv',header=True,index=False)
days_info = {'length_day':[lenday],'number_days':[lenfiles],'number_test_days':[lenfiles_test],'number_train_days':[lenfiles_train],'seed_used':[ran_seed]}
days_info_csv=pd.DataFrame(days_info)
days_info_csv.to_csv(dest_folder+'/days_info.csv',header=True,index=False)

days_csv = pd.DataFrame({'total_days':input_files})
days_csv.to_csv(dest_folder+'/total_days.csv',header=True,index=False)

days_test_csv = pd.DataFrame({'test_days':input_files_test})
days_test_csv.to_csv(dest_folder+'/test_days.csv',header=True,index=False)

days_train_csv=pd.DataFrame({'train_days':input_files_train})
days_train_csv.to_csv(dest_folder+'/train_days.csv',header=True,index=False)

thresholds_csv = pd.DataFrame({'score':[score_to_use],'threshold':[scores_threshold]})
thresholds_csv.to_csv(dest_folder+'/threshold_info.csv',header=True,index=False)

end_time = time.time()
print('{} files appended in {} seconds. Splitting matrices.\n'.format(lenfiles_train,end_time-start_time))


#And now we DO normalize. This is far from optimal since we load what we just saved and closed...

az_range = az_max - az_min
ele_range = ele_max - ele_min

X_test = pd.read_csv(test+'/X_test.csv')
X_test.elevation = ((X_test.elevation - ele_min) / ele_range).round(5)
X_test.azimut = ((X_test.azimut - az_min) / az_range).round(5)
X_test.to_csv(test+'/X_test.csv',header=True,index=False)

X_train = pd.read_csv(tv+'/X_tr_val.csv')
X_train.elevation = ((X_train.elevation - ele_min) / ele_range).round(5)
X_train.azimut = ((X_train.azimut - az_min) / az_range).round(5)
X_train.to_csv(tv+'/X_tr_val.csv',header=True,index=False)
