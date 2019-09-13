#Cross correlation to determine which stations to choose
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import optparse
import json
import os

import datetime
import copy

def addOptions(parser):
    parser.add_option("--config", default="",
            help="confguration file")

def computePositive(data):
    threshold_y = 0.01
    threshold_x = 91#901
    for i in range(91,179):
#    for i in range(901,1799):
        if data[i] < threshold_y and data[threshold_x] >= threshold_y:
            threshold_x = i 
#    return threshold_x, data[901:threshold_x].sum()
    return threshold_x, data[91:threshold_x].sum()

parser = optparse.OptionParser()
addOptions(parser)

(options, args) = parser.parse_args()

if not options.config:
    print >> sys.stderr, "No configuration file specified\n"
    sys.exit(1)


#Stations to consider
stations = ['dh3', 'dh4', 'dh5', 'dh10', 'dh11', 'dh9', 'dh2', 'dh1',
            'ap6', 'ap1', 'ap3', 'ap5', 'ap4', 'ap7', 'dh6', 'dh7', 'dh8']

with open(options.config, 'r') as cfg_file:
    cfg_data = json.load(cfg_file)

orig_folder = cfg_data['orig_folder']
dest_folder = cfg_data['dest_folder']
stations = cfg_data['stations']
#reverse order, only to run, since encountered an out of memory error killing the process before finishing its work
#stations = stations[::-1]
ini_plt = 649#879#0#835#879#839#0#839
end_plt = -649#-880#-1#-835#-880#-840#-1#-840
vers = 21


npts = 3600 # number of seconds in each hour
thrs = 0#0.85 # threshold value to add up fot the hist
step = 900# step to shift the treatment window of the data 15 min = 15x60 seconds
step = int(step/10)
pts = 40

medias = []
desv_tip = []
hists = []
secs = []
perc = []
perc_dev = []
med_dev = []
max_dev = []
stat_cols = ['histogram', 'mean', 'standard_deviation', 'max_number_sec', 'percentage_of_data', 'area_std_dev', 'mean_std_dev', 'max_std_dev']

####################################

# time limits
mor_beg = datetime.datetime.strptime("07:30","%H:%M")
mor_mid = datetime.datetime.strptime("11:00","%H:%M")
mid_aft = datetime.datetime.strptime("14:00","%H:%M")
aft_end = datetime.datetime.strptime("17:30","%H:%M")


months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

hours = ['full', 'morning', 'midday', 'afternoon']

seas = ['year', 'winter', 'spring', 'summer', 'autumn']
counts = {}
h = {}
for i in hours:
    h[i] = 0
for i in months:
    counts[i] = copy.deepcopy(h)
for i in seas:
    counts[i] = 0
counts['cur'] = 0
counts['step'] = [0 for x in range(pts)]

count = {}
inside = {}
for i in hours:
    inside[i] = [0 for x in range(-step+1,step)]
for i in months:
    count[i] = copy.deepcopy(inside)
for i in seas:
    count[i] = [0 for x in range(-step+1,step)]
count['cur'] = [0 for x in range(-step+1,step)]
count['step'] = [[0 for x in range(-step+1,step)] for y in range(pts)]
####################################
lags = np.arange(-step+1, step)

for sta in stations:
    curr_path = orig_folder + '/' + sta + '/'
    input_files = [f for f in os.listdir(curr_path) if f.endswith('.csv')]
    sta_dates = set([date[:8] for date in input_files])
    for stas in stations:
        path = orig_folder + '/' + stas + '/'
        dates = [f[:8] for f in os.listdir(path) if f[:8] in sta_dates]
        print('Between %s with %d and %s with %d there are %d same dates'%(str(sta),len(sta_dates),str(stas),len([f for f in os.listdir(path) if f.endswith('.csv')]),len(dates)))
        ## Reset variables for histograms
        for i in hours:
            h[i] = 0
        for i in months:
            counts[i] = copy.deepcopy(h)
        for i in seas:
            counts[i] = 0
        counts['cur'] = 0
        counts['step'] = [0 for x in range(pts)]

        # histograms
        count = {}
        inside = {}
        for i in hours:
            inside[i] = [0 for x in range(-step+1,step)]
        for i in months:
            count[i] = copy.deepcopy(inside)
        for i in seas:
            count[i] = [0 for x in range(-step+1,step)]
        count['step'] = [[0 for x in range(-step+1,step)] for y in range(pts)]


        for input_file in dates:
            # set to 0 for the date
            count['cur'] = [0 for x in range(-step+1,step)]
            counts['cur'] = 0
            df_target = pd.read_csv(curr_path + input_file + '_' + sta + '.csv', engine='python')
            df = pd.read_csv(path + input_file + '_' + stas + '.csv', engine='python')
            iter = len(df) - npts
            index = 0

            time_granularity = 10
            decimal_pos = 5
            ghi_means = []
            ghi_means2 = []
            samp = int(len(df) / time_granularity)
            gr = df_target[df_target.columns[-1]].values
            grs = df[df.columns[-1]].values
            for index in range(samp):
                a = round(gr[index*time_granularity:(index+1)*time_granularity].mean(),decimal_pos)
                b = round(grs[index*time_granularity:(index+1)*time_granularity].mean(),decimal_pos)
                ghi_means.append(a)
                ghi_means2.append(b)
            _df_target = pd.DataFrame(ghi_means)
            _df = pd.DataFrame(ghi_means2)
            iter = int(len(_df) - step)
            index = 0

            for i in range(0,iter,step):
                sta_range = _df_target[i:(i+step)]
                stas_range = _df[i:(i+step)]
                sta_data = sta_range[sta_range.columns[-1]]
                stas_data = stas_range[stas_range.columns[-1]]
                hor = np.linspace(i, i+step, step)
                ccov = np.correlate(sta_data-sta_data.mean(), stas_data-stas_data.mean(), mode='full')
                ccor = ccov/(step * sta_data.std() * stas_data.std())
                maximum = np.argmax(ccor)
                
                if np.max(ccor) >= thrs:
                    #filter the crosscorreltaion
                    maxs = np.where(ccor < thrs, 0, ccor)
                    # whole data (1,5 year aprox.)
                    count[seas[0]] += maxs
                    counts[seas[0]] += 1
                    # seasons
                    quarter = 0
                    today = int(input_file[4:8])
                    if today > 320 and today < 621:
                        quarter = 2
                    elif today > 620 and today < 921:
                        quarter = 3
                    elif today > 920 and today < 1221:
                        quarter = 4
                    else:
                        quarter = 1
                    count[seas[quarter]] += maxs
                    counts[seas[quarter]] += 1
                    # months
                    month = int(input_file[4:6])-1
                    _mor_mid = (mor_mid - mor_beg) / 10
                    _mid_aft = (mid_aft - mor_beg) / 10
                    interval = 0
                    if i < _mor_mid.seconds:
                        interval = 1
                    elif i >= _mor_mid.seconds and i < _mid_aft.seconds:
                        interval = 2
                    elif i >= _mid_aft.seconds:
                        interval = 3
                    count[months[month]][hours[interval]] += maxs
                    counts[months[month]][hours[interval]] += 1
                    count[months[month]][hours[0]] += maxs
                    counts[months[month]][hours[0]] += 1
                    # day
                    counts['cur'] += 1
                    count['cur'] += maxs
                    # times
                    counts['step'][index] += 1
                    count['step'][index] += maxs
                index = index + 1

############ day ############
            count_cur = np.array(count['cur'])
            if counts['cur'] != 0:
                count_cur = count_cur / counts['cur']
            if np.count_nonzero(count_cur) == 0:
                mean = 0
                std_var = 0
            else:
                mean = np.average(lags, weights=count_cur)
                var = np.average((lags-mean)**2, weights=count_cur)
                std_var = np.sqrt(var)
            medias.append(mean)
            desv_tip.append(std_var)
            max_sec, part = computePositive(count_cur)
            secs.append(max_sec)
            perc.append(part)
            ini_ind = int(step+mean-std_var-1)
            end_ind = int(step+mean+std_var)
            perc_dev.append(count_cur[ini_ind:end_ind].sum())
            med_dev.append(np.mean(count_cur[ini_ind:end_ind]))
            max_dev.append(np.max(count_cur[ini_ind:end_ind]))
            hists.append(sta + '_' + stas +'_'+input_file+ '_threshold')
######## Seasons ########
        for seasons in seas:
            _array = np.array(count[seasons])
            if counts[seasons] != 0:
                _array = _array / counts[seasons]
            if np.count_nonzero(_array) == 0:
                mean = 0
                std_var = 0
            else:
                mean = np.average(lags, weights=_array)
                var = np.average((lags-mean)**2, weights=_array)
                std_var = np.sqrt(var)
            medias.append(mean)
            desv_tip.append(std_var)
            max_sec, part = computePositive(_array)
            secs.append(max_sec)
            perc.append(part)
            ini_ind = int(step+mean-std_var-1)
            end_ind = int(step+mean+std_var)
            perc_dev.append(_array[ini_ind:end_ind].sum())
            med_dev.append(np.mean(_array[ini_ind:end_ind]))
            max_dev.append(np.max(_array[ini_ind:end_ind]))
            hists.append(sta + '_' + stas +'_' + seasons)
######## months ########
        for m in months:
            for hrs in hours:
                _array = np.array(count[m][hrs])
                if counts[m][hrs] != 0:
                    _array = _array / counts[m][hrs]
                if np.count_nonzero(_array) == 0:
                    mean = 0
                    std_var = 0
                else:
                    mean = np.average(lags, weights=_array)
                    var = np.average((lags-mean)**2, weights=_array)
                    std_var = np.sqrt(var)
                medias.append(mean)
                desv_tip.append(std_var)
                max_sec, part = computePositive(_array)
                secs.append(max_sec)
                perc.append(part)
                ini_ind = int(step+mean-std_var-1)
                end_ind = int(step+mean+std_var)
                perc_dev.append(_array[ini_ind:end_ind].sum())
                med_dev.append(np.mean(_array[ini_ind:end_ind]))
                max_dev.append(np.max(_array[ini_ind:end_ind]))
                hists.append(sta + '_' + stas +'_' + m + '_' + hrs)

        # Stats over data (mean)
        more_stats = {}
        summary = ['mean', 'sitd_dev', 'secs', 'perc', 'perc_dev', 'med_dev', 'max_dev']
        for smry in summary:
            more_stats[smry] = {}
            for hrs in hours:
                more_stats[smry][hrs] = []
        for i in range(len(hists)):
            for hrs in hours:
                if hists[i][(len(hists[i])-len(hrs)):] == hrs:
                    more_stats[summary[0]][hrs].append(medias[i])
                    more_stats[summary[1]][hrs].append(desv_tip[i])

                    more_stats[summary[2]][hrs].append(secs[i])
                    more_stats[summary[3]][hrs].append(perc[i])
                    more_stats[summary[4]][hrs].append(perc_dev[i])
                    more_stats[summary[5]][hrs].append(med_dev[i])
                    more_stats[summary[6]][hrs].append(max_dev[i])
        for hrs in hours:
            medias.append(np.mean(more_stats[summary[0]][hrs]))
            desv_tip.append(np.mean(more_stats[summary[1]][hrs]))
            secs.append(np.mean(more_stats[summary[2]][hrs]))
            perc.append(np.mean(more_stats[summary[3]][hrs]))
            perc_dev.append(np.mean(more_stats[summary[4]][hrs]))
            med_dev.append(np.mean(more_stats[summary[5]][hrs]))
            max_dev.append(np.mean(more_stats[summary[6]][hrs]))
            hists.append(sta + '_' + stas + '_' + hrs + 'mean')
 
######## times ########
        for i in range(pts):
            np_array = np.array(count['step'][i])
            if counts['step'][i] != 0:
                np_array = np_array / counts['step'][i]
            if np.count_nonzero(np_array) == 0:
                mean = 0
                std_var = 0
            else:
                mean = np.average(lags, weights=np_array)
                var = np.average((lags-mean)**2, weights=np_array)
                std_var = np.sqrt(var)
            medias.append(mean)
            desv_tip.append(std_var)
            max_sec, part = computePositive(np_array)
            secs.append(max_sec)
            perc.append(part)
            ini_ind = int(step+mean-std_var-1)
            end_ind = int(step+mean+std_var)
            perc_dev.append(np_array[ini_ind:end_ind].sum())
            med_dev.append(np.mean(np_array[ini_ind:end_ind]))
            max_dev.append(np.max(np_array[ini_ind:end_ind]))
            hists.append(sta + '/' + sta + '_' + stas +'_'+ str(i))

        _mean = pd.DataFrame(medias)
        _std_var = pd.DataFrame(desv_tip)
        _secs = pd.DataFrame(secs)
        _perc = pd.DataFrame(perc)
        _perc_dev = pd.DataFrame(perc_dev)
        _med_dev = pd.DataFrame(med_dev)
        _max_dev = pd.DataFrame(max_dev)
        _hists = pd.DataFrame(hists)
        matrix = pd.concat([_hists,_mean.reset_index(drop=True),_std_var.reset_index(drop=True), _secs.reset_index(drop=True), _perc.reset_index(drop=True), _perc_dev.reset_index(drop=True), _med_dev.reset_index(drop=True), _max_dev.reset_index(drop=True)], axis=1)
        matrix.columns = stat_cols
        matrix.to_csv(dest_folder + '/' + sta + '/histogram_stats'+ str(vers) +'.csv', header=True, index=False)

