import datetime
from db import dbfunctions as db
from tracker import models
import numpy as np
import scipy
import scipy.io as sio
#import sio_adjustment as sio
import sys
import simple_spec as ss
import tables
import multiprocessing
import gc #garbage collector

#storage_path = '/home/helene/Downloads/pk/'

def update_task_entry_list(task_name='manual_control_multi_plusvar', task_type='S1'):
    current_status = sio.loadmat(storage_path+'_'+task_type+'_status.mat',)
    old_te = current_status['task_entries']
    te = get_ids(task_name)
    dic = filter_into_behav_states(te)
    prep = task_type[0]
    tar = int(task_type[1])
    all_task_entries = set(dic[prep])&set(dic[tar])
    new_task_entries = all_task_entries - set(old_te)
    return list(new_task_entries)

def update_trials_and_power(new_task_entries, save=True, task_type = 'S1'):
    power_dict, trial_dict, current_status = extract_all_trials(save=False,task_entries=new_task_entries)

    P = sio.loadmat(storage_path+'_'+task_type+'_power_dict.mat')
    T = sio.loadmat(storage_path+'_'+task_type+'_trial_dict.mat')
    S = sio.loadmat(storage_path+'_'+task_type+'_status.mat')

    prep = task_type[0]
    tar = int(task_type[1])
 
    #Add to status first: 
    S['n',task_type] = S['n',task_type].extend(current_status['n',task_type])
    S['task_entries',task_type] = S['task_entries',task_type].extend(current_status['task_entries',task_type])

    #Now add to trials: 
    T[task_type] = np.vstack(( T[task_type], trial_dict[task_type] ))

    #Now Power (by channel)
    for c, chan in enumerate(P['channels'][0,:]):
        P[task_type, chan] = np.vstack(( P[task_type, chan], power_dict[task_type,chan]))

    if save:
        sio.savemat(storage_path+'_'+task_type+'_power_dict.mat',P)
        sio.savemat(storage_path+'_'+task_type+'_trial_dict.mat',trials_dict)
        sio.savemat(storage_path+'_'+task_type+'_status.mat',current_status)
    else:
        return P, T, S

# def extract_behav_metrics(task_ts, hdf, generator = 'centerout'):
#     '''
#     Function to extract:
#         center hold error rate, 
#         center hold times,
#         reach accuracy,
#         reach time,
#         trials per minute
#         rewards per minute
#     '''

#     # #Check all correspond to 'targ_transition'
#     # for i, g in enumerate(go_ind):
#     #     if msgs[g][0] != np.string_('targ_transition'):
#     #         print 'error -- task structure not correct' + str(i) + ', ' +msg[g][0]
#     #     elif msgs[g-3][0] != np.string_('wait'):
#     #         print 'error trial'
#     #         #If there's a hold error or timeout error, ignore this trial
#     #     else:
#     #         filt_go_ind.extend([g])


#     # task_ts = hdf.root.task_msgs[filt_go_ind]['time']

def get_plx_with_hdf_inds(task_type, te_num, hdf_inds, channels=[124, 126, 252, 154, 2], 
    dbname = 'default',**kwargs):
    if 't_range' in kwargs.keys():
        t_range = kwargs['t_range']
    else:
        t_range = [.5, 3.5]

    task = db.TaskEntry(te_num,dbname=dbname)
    nm = task.name

    if 'system' in kwargs.keys():
        files_ok, plx, hdf, ts_func = load_session(nm,system=kwargs['system'])
    else:
        files_ok, plx, hdf, ts_func = load_session(nm)

    if 'get_neural_only' in kwargs.keys():
        get_neural_only = kwargs['get_neural_only']
    else:
        get_neural_only = False

    if files_ok: 
        plx_ts = ts_func(hdf_inds,'plx')
        trials, channels = get_trials(plx_ts, plx,t_range=t_range, channels=channels)

        if 'small_f_steps' in kwargs.keys():
            kwargs['small_f_steps'] = True
        kwargs['type'] = kwargs['spec_method']
        power_dict = dict()

        if get_neural_only:
            print 'skipping PXX'
            Pxx=dict(bins=0, freq=0)
        else:
            Pxx = get_power(trials, channels, **kwargs)
            for c,chan in enumerate(channels):
                power_dict[task_type,chan] = Pxx[str(chan)]
                
            power_dict['bins'] = Pxx['bins']
            power_dict['freq'] = Pxx['freq']
            power_dict['channels'] = Pxx['chan_ord']
        return files_ok, power_dict, trials, channels, Pxx['bins'], Pxx['freq']
    else: 
        return files_ok, 0, 0, 0, 0, 0
    

def extract_all_trials(task_type, data_type, save_ind_te, task_entry, channels,
    manual_task_name='manual_control_multi_plusvar',storage_path='/home/preeya/Cart/',
    dbname='default'):

    power_dict = dict()
    trials_dict = dict() # trials x time x channel
    current_status = dict()

    t = task_entry #integer. 
    print 'Task Entry: ', str(t)
    task = db.TaskEntry(t,dbname=dbname)
    nm = task.name
    targ_idx = 2 # Out trials 

    task_ts, plx_ts, hdf, plx = get_go_cue_idx(nm, targ_idx)

    #Plex Data
    if data_type == 'plx':
        trials, channels = get_trials(plx_ts, plx,channels=channels)

    elif data_type == 'cursor':
        trials = get_cursor(task_ts, hdf)

    elif data_type == 'targ_loc':
        trials = get_targ_loc(task_ts, hdf)

    elif data_type == 'behavior':
        trial_behav = extract_behav_metrics(data_type, task_ts, hdf)

    print 'shape of trials: ', trials.shape

    trials_dict[task_type] = trials
    current_status['n',task_type] = np.array([trials.shape[0]])
            
    if data_type == 'targ_loc':
        trials_dict['go_cue_ind'] = task_ts
        trials_dict['n'] = len(task_ts)

    if data_type == 'plx':
        Pxx = get_power(trials, channels)
        power_dict = dict()
        
        for c,chan in enumerate(channels):
            power_dict[task_type,chan] = Pxx[str(chan)]
        
        power_dict['bins'] = Pxx['bins']
        power_dict['freq'] = Pxx['freq']
        power_dict['channels'] = Pxx['chan_ord']

    i = 0
    if save_ind_te:
        if data_type == 'plx':
            sio.savemat(storage_path+'mc_files/cart_pwr/'+'tmp_pwr_'+str(i)+'_'+str(t)+'_'+task_type+'.mat', power_dict)
            sio.savemat(storage_path+'mc_files/cart_pwr/'+'tmp_neural_signal_trials_'+str(i)+'_'+str(t)+'_'+task_type+'.mat', trials_dict)
        
        elif data_type == 'cursor':
            sio.savemat(storage_path+'mc_files/cart_cursor/'+'tmp_trials_'+str(i)+'_'+str(t)+'_'+task_type+'.mat', trials_dict)
        
        elif data_type == 'targ_loc':
            sio.savemat(storage_path+'mc_files/cart_targ/'+'tmp_targ_locs_'+str(i)+'_'+str(t)+'_'+task_type+'.mat', trials_dict)
            
        sio.savemat(storage_path+'_'+task_type+'_status.mat',current_status)


def get_ids(task_name,startdate=datetime.date(2014,10,26),enddate=datetime.date.today,
    min_task_len=None,system='sdh', dbname = 'default', **kwargs):
    
    '''Get relevant task IDs with plexon files associated with them'''
    #mcm_id = db.get_task_id(task_name)
    #task_list = models.TaskEntry.objects.filter(task__name=task_name).filter(date__gte=startdate).filter(date__lte=enddate)
    task_list = models.TaskEntry.objects.using(dbname).filter(task__name=task_name).filter(date__gte=startdate).filter(date__lte=enddate)

    #Hack, set dbpath:
    if dbname == 'default':
        dbpath = 'bmi3d/rawdata/hdf/'
    elif dbname == 'exorig':
        dbpath = 'exorig/rawdata/hdf/'
    else:
        print 'Unrecognized database!'

    if system == 'sdh':
        storage_path = '/storage/'

    elif system == 'pk_mbp':
        storage ='/Volumes/carmena/'

    elif system == 'nucleus':
        dbpath = 'rawdata/hdf/'
        storage_path = '/storage/'
        
    else:
        print 'Unrecognized system!'

    te = []

    for i,tsk_entry in enumerate(task_list):
        add_task = 1
        #Use TaskEntry from dbfunctions:
        tsk_id = tsk_entry.id
        try:
            task_entry = db.TaskEntry(tsk_id,dbname=dbname)
            task_acquired = 1
        except:
            print "Cannot get task: ", tsk_id
            task_acquired = 0
        
        #Make sure of plx file
        if task_acquired:     
            if len(task_entry.plx_filename) > 0:
                add_task = np.min([1, add_task])
            else:
                add_task = np.min([0, add_task])
                print 'No plx file', task_entry

            #Minimum length criterion?

            if min_task_len is not None and add_task:
                try:
                    hdf = tables.openFile(storage_path+dbpath+task_entry.name+'.hdf')
                    hdf_file_acquired = 1
                except:
                    print 'No HDF file:', storage_path+dbpath+task_entry.name+'.hdf', task_entry
                    hdf_file_acquired = 0
                    add_task = np.min([0, add_task])

                try:
                    tm = hdf.root.task_msgs
                except:
                    print 'HDF File corrupt, no task_msgs'
                    hdf_file_acquired = 0
                    add_task = np.min([0, add_task])
                    
            if add_task and hdf_file_acquired:
                sess_len_minutes= hdf.root.task_msgs[-1]['time']/(60.*60.)
                
                if sess_len_minutes>= min_task_len:
                    add_task = np.min([1, add_task])
                else: 
                    add_task = np.min([0, add_task])

            if 'te_max' in kwargs.keys() and add_task and kwargs['te_max'] is not None:
                if task_entry.id <= kwargs['te_max']:
                    add_task = np.min([1, add_task])
                else:
                    add_task = np.min([0, add_task])
    
            if 'te_min' in kwargs.keys() and add_task and kwargs['te_min'] is not None:
                if task_entry.id >= kwargs['te_min']:
                    add_task = np.min([1, add_task])
                else:
                    add_task = np.min([0, add_task])

            if add_task: 
                te.extend([task_entry.id])
    return te

def filter_into_behav_states(te,pref='/storage/rawdata/hdf/',min_session_length=5,dbname='default'):
    d = dict()
    d['S'] = []
    d['M'] = []
    d['L'] = []

    d[1] = []
    d[4] = []
    d[8] = []
    

    for i, t in enumerate(te):
        task = db.TaskEntry(t, dbname=dbname)

        #Ensure there's a plexon file attached (double checked, also in get_ids)
        if (len(task.plx_filename) > 0) and (task.length > (min_session_length*60)):

            #Split into S vs. M vs. L
            if task.params['hold_variance'] < 0.1:
                d['S'].extend([t])
            elif task.params['hold_time'] > 1:
                d['L'].extend([t])
            else:
                d['M'].extend([t])

            #Find number of unique targets
            nm = pref+task.name+'.hdf'
            hdf = tables.openFile(nm)
            tg = hdf.root.task[:]['target']
            tg2 =  np.ascontiguousarray(tg).view(np.dtype((np.void, tg.dtype.itemsize * tg.shape[1])))
            tg_num,idx = np.unique(tg2,return_index=True)
            unique_tgs = tg[idx]

            if unique_tgs.shape[0] == 9:
                d[8].extend([t])
            elif unique_tgs.shape[0] == 5:
                d[4].extend([t])
            elif unique_tgs.shape[0] == 2:
                d[1].extend([t])
            else: 
                print 'unknown number of targets!'
    return d

def get_go_cue_idx(task_name,target_num=2,generator='centerout'):
    '''Target_num is 2 for center out reaches, or 3 for center in reaches
       t_range is tuple corresponding to time before and time after go_cue'''
    files_ok, plx, hdf, ts_func = load_session(task_name)
    msgs = hdf.root.task_msgs[:]
    rew_ind = np.array([i for i, m in enumerate(msgs) if m[0] == 'reward'])

    if generator is 'centerout_and_back':
        if target_num == 2:
            go_ind = rew_ind - 7
        elif target_num == 3:
            go_ind = rew_ind - 4
        else:
            print 'error -- not specified appropriate target_num (must be 2 or 3)'
    
    elif generator is 'centerout':
        if target_num ==2:
            go_ind = rew_ind - 4

    filt_go_ind = []
    
    #Check all correspond to 'targ_transition'
    for i, g in enumerate(go_ind):
        if msgs[g][0] != np.string_('targ_transition'):
            print 'error -- task structure not correct' + str(i) + ', ' +msg[g][0]
        elif msgs[g-3][0] != np.string_('wait'):
            print 'error trial'
            #If there's a hold error or timeout error, ignore this trial
        else:
            filt_go_ind.extend([g])


    task_ts = hdf.root.task_msgs[filt_go_ind]['time']
    plx_ts = ts_func(task_ts,'plx')
    return task_ts, plx_ts, hdf, plx

def get_CHE_error_times(task_name, target_num=2):
    hdf = load_session(task_name, hdf_only=True)
    msgs = hdf.root.task_msgs[:]

def get_trials(plx_ts, plx ,t_range = [2.5, 1.5], Plx_fs=1000, channels = range(0,257,32)):
    ''' Returns trials x time x channel''' 
    trials = np.zeros((  len(plx_ts), int(Plx_fs*(t_range[0] + t_range[1])), len(channels) ))
    
    for i, g in enumerate(plx_ts):
        begid = g - t_range[0]
        endid = g + t_range[1]
        try:
            trials[i,:,:] = np.reshape(plx.lfp[begid:endid,channels].data, (1, trials.shape[1], len(channels)) )
        except:
            tmp = plx.lfp[begid:endid,channels].data
            if tmp.shape[0] == 3501:
                tmp = tmp[1:]
                print 'shaved by 1 ms'
            trials[i,:tmp.shape[0],:] = np.reshape(tmp, (1, tmp.shape[0], len(channels)) )
    return trials, channels

def get_targ_loc(task_ts, hdf):
    '''Returns trials x 3'''

    trials = np.zeros(( len(task_ts), 2))
    for i, g in enumerate(task_ts):
        trials[i,:] = hdf.root.task[int(g)+5]['target'][[0,2]]
    return trials

def get_cursor(task_ts, hdf, t_range= [2.5, 1.5], Task_fs =60):
    trials = np.zeros((  len(task_ts), Task_fs*(t_range[0] + t_range[1]), 2 ))
    #trial x time x (x,y)
    clip_early = False
    clip_late = False

    for i, g in enumerate(task_ts):
        begid = int(g - (t_range[0]*Task_fs))
        if begid < 0:
            begid = 0
            clip_early = True

        endid = int(g + (t_range[1]*Task_fs))
        if endid > hdf.root.task.shape[0]:
            endid = hdf.root.task.shape[0]
            clip_late = True

        if clip_early:
            trials[i,trials.shape[1]-(endid-begid):,:] = hdf.root.task[begid:endid]['cursor'][:,[0,2]]
        elif clip_late:
            trials[i,:endid-begid,:] = hdf.root.task[begid:endid]['cursor'][:,[0,2]]
        else:
            trials[i,:,:] = hdf.root.task[begid:endid]['cursor'][:,[0,2]]

    return trials


def get_power(trials, channels, type='MTM', moving_window = [0.201, 0.011],small_f_steps = False, **kwargs):
    ''' trials : trials x time x channel '''
    
    if 'bp_filt' in kwargs:
        bp_filt = kwargs['bp_filt']
    else:
        bp_filt = [10, 55]

    Pxx = dict()
    for c, chan in enumerate(channels):
        kw = dict()
        if 'small_f_steps':
            kw['small_f_steps'] = True

        if type == 'MTM':
            S, f, t = ss.MTM_specgram(trials[:,:,c].T, movingwin=moving_window,**kw)
            
        elif type == 'DFT':
            S, f, t = ss.DFT_PSD(trials[:,:,c].T,movingwin=moving_window)
        
        elif type == 'Welch':

            S, f, t = ss.Welch_specgram(trials[:,:,c].T, movingwin=moving_window, bp_filt=bp_filt)

        
        Pxx[str(chan)] = S
    Pxx['bins'] = t
    Pxx['freq'] = f
    Pxx['chan_ord'] = channels
    return Pxx


#######
#######
# From analysis/basicanalysis.py
#######
#######

import numpy as np
from plexon import plexfile, psth
import tables
from riglib.dio import parse
from pylab import specgram
import os.path

def load_session(session_name,hdf_only=False,system='sdh',dbname='default'):

    #plx_path = '/Volumes/carmena/bmi3d/plexon/'
    #hdf_path = '/Volumes/carmena/bmi3d/rawdata/hdf/'
    #binned_spikes_path = '/Volumes/carmena/bmi3d/binned_spikes/'
    if system is 'sdh' and dbname is 'default':
        plx_path = '/storage/bmi3d/plexon/'
        hdf_path = '/storage/bmi3d/rawdata/hdf/'
    elif system is 'sdh' and dbname is 'exorig':
        plx_path = '/storage/exorig/plexon/'
        hdf_path = '/storage/exorig/rawdata/hdf/'        
    elif system in ['arc','nucleus']:
        plx_path = '/storage/plexon/'
        hdf_path = '/storage/rawdata/hdf/'
    elif system in ['arc_backup']:
        plx_path = '/backup/exorig/plexon/'
        hdf_path = '/backup/exorig/rawdata/hdf/'

    '''

    Load all files associated with a recording session and extract timestamps.

    Parameters
    ----------
    session_name : string
        The name of the session of interest without file extension.

    Returns
    -------
    plx : plexon file
        The loaded plexon file.
    hdf : hdf file
        The loaded hdf5 file.
    ts_func : function
        A function that translates plexon timestamps to hdf row indices or vice
        versa for this session.

        Parameters:
        input_times : list of either plx timestamps (floats) or hdf timestamps
        (ints) to translate
        output_type : string ['hdf', 'plx'] specifying which type the output
        should be (should NOT be the same as the input type)

        Returns:
        output : list of either plx or hdf timestamps corresponding to input


    '''
    hdf = tables.openFile(hdf_path + session_name + '.hdf')
    
    if not hdf_only:
        plx = plexfile.openFile(plx_path + session_name + '.plx')
    
        def sys_eq(sys1, sys2):
            return sys1 in [sys2, sys2[1:]]

        events = plx.events[:].data
        # get system registrations
        reg = parse.registrations(events)
        syskey = None

        # find the key for the task data
        for key, system in reg.items():
            if sys_eq(system[0], 'task'):
                syskey = key
                break

        if syskey is None: 
            print 'NO SYSKEY Error'
            files_ok = False
            plx = ts_func =0
        
        else:
            ts = parse.rowbyte(events)[syskey] 

            # Use checksum in ts to make sure there are the right number of rows in hdf.
            if len(hdf.root.task)<len(ts):
                ts = ts[1:]
        
            if np.all(np.arange(len(ts))%256==ts[:,1]):
                print "Dropped frames detected!"
                files_ok = True 

            files_ok = True 
            if len(ts) < len(hdf.root.task):
                print "Warning! Frames missing at end of plx file. Plx recording may have been stopped early."

            ts = ts[:,0]

            # Define a function to translate plx timestamps to hdf and vice versa for
            # this session.
            def ts_func(input_times, output_type):

                if output_type == 'plx':
                    if len(input_times)>len(ts):
                        input_times = input_times[:len(ts)]
                    output = [ts[time] for time in input_times]

                if output_type == 'hdf':
                    output = [np.searchsorted(ts, time) for time in input_times]

                return np.array(output)

        # Check for previously saved binned spike file, save one if doesn't exist
        #filename = binned_spikes_path+session_name
        #if not os.path.isfile(filename+'.npz'):
        #    save_binned_spike_data(plx, hdf, ts_func, filename)

        return files_ok, plx, hdf, ts_func
    else:
        return hdf

if __name__ == '__main__':
    #task = ['S1', 'S4','M1','M4','L1','L4']
    #for t, task_type in enumerate(task):
    task_type = sys.argv[1]
    print task_type
    te = sio.loadmat('/home/helene/Downloads/pk/te_num_file.mat')
    entries = te[task_type][0,:]
    jobs = []
    for i, e in enumerate(entries):
        #p = multiprocessing.Process(target=extract_all_trials, args = (task_type, 'plx', True, [e]))
        #jobs.append(p)
        #p.start()
        #print 'starting ', task_type, str(e), ': plx'
        #p1 = multiprocessing.Process(target=extract_all_trials, args = (task_type, 'cursor', True, [e]))
        p1 = multiprocessing.Process(target=extract_all_trials, args = (task_type, 'targ_loc', True, [e]))
        jobs.append(p1)
        p1.start()
        print 'starting ', task_type, str(e), ': cursor'