# Used for task: manual_control_multi_plus_var

from tables import *
import tables
from sklearn.preprocessing import normalize
import numpy as np
import psycho_metrics as pm
import scipy.io as sio
import dbfunctions as dbfn
import fcns
import extract_cart_sdh as ecsdh
import multiprocessing
import gc
import datetime
import mc_metrics

class Behav_Hand_Reach(IsDescription):
    trial_type = StringCol(2)
    trial_outcome = StringCol(14)
    target_loc = Float64Col(shape=(2,))
    reach_time = Float64Col()
    rxn_time = Float64Col()
    go_time = Float64Col()
    hold_time = Float64Col()
    reach_err = Float64Col()
    rew_rate = Float64Col()
    start_time = Float64Col()
    task_entry = IntCol()

class Neural_Hand_Reach(IsDescription):
    trial_type = StringCol(2)
    task_entry = IntCol()
    start_time = IntCol()
    #neural_sig = Float32Col(shape=(3500,5)) #time x channels
    power_sig = Float32Col(shape =(296,26)) #time x freq x channel
    #long_neural_sig = Float32Col(shape=(5500, 5))
    long_power_sig = Float32Col(shape = (409, 103))

class Neural_Hand_Reach_neur_only(Neural_Hand_Reach):
    neural_sig = Float32Col(shape=(3500,5)) #time x channels
    power_sig = IntCol()
    long_neural_sig = Float32Col(shape=(5500, 5))
    long_power_sig = IntCol()


class Neural_Hand_Reach_small_f_step(Neural_Hand_Reach):
    trial_type = StringCol(2)
    task_entry = IntCol()
    start_time = IntCol()
    neural_sig = Float32Col(shape=(3500)) #time x channels
    power_sig = Float32Col(shape =(300,26)) #time x freq x channel

class Kin_Traces(IsDescription):
    cursor_traj = Float64Col(shape=(int(2000*(60./1000)), 2))
    mc_vect = Float64Col(shape=2,)
    kin_sig = Float64Col(shape=(int(2000*(60./1000))-1, ))
    kin_feat = Float64Col(shape=5, )
    task_entry = IntCol()
    start_time = IntCol()

class manual_control_data(object):

    def __init__(self, *args, **kwargs):
        self.behav_fname = kwargs['behav_file_name']
        self.neur_fname = kwargs['neural_file_name']
        self.task_entry_dict_fname = kwargs['task_entry_dict_fname']
        self.task_entry_dict = sio.loadmat(self.task_entry_dict_fname)
        self.task_entry_dict_time_inds_fname = self.task_entry_dict_fname[:-3]+'_start_inds.h5'
        self.task_entry_dict_go_times_fname = kwargs['task_entry_dict_go']+'.mat'
        if 'tasks' in kwargs.keys():
            self.tasks = kwargs['tasks']
        else:
            self.tasks = ['S1','S4','M1','M4']

        if 'system' in kwargs.keys():
            self.system = kwargs['system']
        else:
            self.system = 'sdh'

        self.tdy_str = datetime.date.today().isoformat()
        self.spec_method = kwargs.pop('spec_method', 'MTM')
        self.moving_window = kwargs.pop('moving_window', [.201, .011])
        self.neural_sig_only = kwargs.pop('neural_sig_only', False)


    def make_neural(self, task_name, **kwargs):
        task_entry_dict = self.task_entry_dict #defined in init

        if 'use_go_file' in kwargs.keys():
            task_entry_plus_start_inds_dict = sio.loadmat(self.task_entry_dict_go_times_fname)
            str_flag = True

        else:
            if hasattr(self,'task_entry_dict_time_inds'):
                task_entry_plus_start_inds_dict = self.task_entry_dict_time_inds #defined in behav
                str_flag = False
            else:
                try:
                    task_entry_plus_start_inds_dict = sio.loadmat(self.task_entry_dict_time_inds_fname)
                    str_flag = True
                except:
                    print 'Run behavior to get start inds!'

        if 'small_f_steps' in kwargs.keys():
            self.neur_fname = self.neur_fname + 'small_f_steps'

        if 't_range' in kwargs:
            t_range = kwargs['t_range']
        else:
            t_range = [0.5, 3.5]

        if t_range[0]+t_range[1] > 3.5:
            long_trials = True
            self.bp_filt = [.1, 55]
        else:
            long_trials = False
            self.bp_filt = [10, 55]

        get_neural_only = self.neural_sig_only

        bad_te = []
        for tsk in task_name:
            h5file = openFile(self.tdy_str + self.neur_fname + tsk + '.h5', mode="w", title='Cart, neural')
            if 'small_f_steps' in kwargs.keys():
                table = h5file.createTable("/", 'neural', Neural_Hand_Reach_small_f_step, "Neural Table")
            else:
                if get_neural_only:
                    table = h5file.createTable("/", 'neural', Neural_Hand_Reach_neur_only, "Neural Table")
                else:
                    table = h5file.createTable("/", 'neural', Neural_Hand_Reach, "Neural Table")
            
            t1, t2 = task_entry_dict[tsk].shape
            if t1==1 and t2==1:
                te_array = np.array([task_entry_dict[tsk][0,0]])
            else:
                te_array = np.squeeze(task_entry_dict[tsk])

            if len(te_array)>0:
                for te in te_array:
                    if str_flag:
                        start_times = np.squeeze(task_entry_plus_start_inds_dict[str((tsk, te))])
                    else:
                        start_times = np.squeeze(task_entry_plus_start_inds_dict[tsk, te])

                    kwargs['spec_method'] = self.spec_method
                    kwargs['system'] = self.system
                    kwargs['moving_window'] = self.moving_window
                    kwargs['bp_filt'] = self.bp_filt
                    kwargs['get_neural_only'] = get_neural_only
                    
                    files_ok, power_dict, trials, channels, bins, freq = ecsdh.get_plx_with_hdf_inds(tsk, te, start_times,**kwargs) 
                    col = gc.collect() #garbage collector

                    if files_ok: 
                        for j in range(len(start_times)):
                            trl = table.row
                            if get_neural_only:
                                tr, tm, ch = trials.shape
                                if long_trials:
                                    trl['long_neural_sig'] = trials[j, :, :]
                                else:
                                    trl['neural_sig'] = trials[j,:,:]

                            else:
                                pxx = np.zeros(( power_dict[(tsk,channels[0])][j,:,:].shape[0], \
                                                 power_dict[(tsk,channels[0])][j,:,:].shape[1], \
                                                 len(channels)
                                                 ))
                                #pxx = np.zeros(((346,20,11)))
                                for ic, c in enumerate(channels):
                                    pxx[:,:,ic] = power_dict[(tsk,c)][j,:,:]

                                tr, tm, ch = trials.shape
                                ptm, pf, pch = pxx.shape
                                f_trim = freq[freq< 100]
                                f_trim_ix = freq<100

                                if long_trials:
                                    #trl['long_neural_sig'] = trials[j, :, :]
                                    trl['long_power_sig'] = pxx[:, f_trim_ix, 0]
                                else:
                                    #trl['neural_sig'] = trials[j,:,:]
                                    trl['power_sig'] = pxx[:, f_trim_ix, 0]
                            
                            trl['trial_type'] = tsk
                            trl['task_entry'] = te
                            trl['start_time'] = start_times[j]
                            trl.append()
                    else:
                        print 'file error, adding to bad_task_entry list!'
                        print 'task: ', tsk, 'task_entry: ', te
                        bad_te.append(te)

                    table.flush()
                if get_neural_only:
                    add_cols = h5file.createGroup(h5file.root, "columns", "Bad TEs, t_range")
                    h5file.createArray(add_cols, 't_range', t_range)
                else:
                    add_cols = h5file.createGroup(h5file.root, "columns", "Channels, Freq, Bins, Bad TEs")
                    h5file.createArray(add_cols, 'channels', np.array(channels))
                    h5file.createArray(add_cols, 'freq', freq)
                    h5file.createArray(add_cols, 'bins', bins)
                    h5file.createArray(add_cols, 't_range', t_range)

                if len(bad_te)>0:
                    h5file.createArray(add_cols, 'bad_task_entries', np.array(bad_te))
                else:
                    h5file.createArray(add_cols, 'bad_task_entries', np.array([-1]))
                h5file.close()

    def get_behavior(self):
        TE = self.task_entry_dict
        TE_start_inds = dict()
        TE_go_inds = dict()

        for i, tsk in enumerate(self.tasks):
            h5file = openFile(self.tdy_str + self.behav_fname + tsk + '.h5', mode="w", title='Cart, behavior')
            table = h5file.createTable("/", 'behav', Behav_Hand_Reach, "Behavior Table")
            kin_table = h5file.createTable("/",'kin',Kin_Traces,"Kin Data")

            te_array = np.squeeze(TE[tsk])

            #Get for ALL trial types reach time, rxn time, hold time, reach error, rew rate, trial type distributions: 
            for j, te in enumerate(te_array):
                task_entry = dbfn.TaskEntry(te)
                nm = task_entry.name
                
                if self.system=='pk_mbp':
                    hdf = tables.openFile('/Volumes/carmena/bmi3d/rawdata/hdf/'+nm+'.hdf')              
                elif self.system == 'sdh':
                    hdf = tables.openFile('/storage/bmi3d/rawdata/hdf/'+nm+'.hdf')              
                elif self.system == 'nucleus':
                    hdf = tables.openFile('/storage/rawdata/hdf/'+nm+'.hdf')
                else:
                    print 'unrecognized system'

                #Trial Starts: 
                msg = hdf.root.task_msgs[:]['msg']
                msg_time = hdf.root.task_msgs[:]['time']
                start_ind = np.array([j for j, i in enumerate(msg) if i=='wait'])
                start_times = msg_time[start_ind]
                rew_times = np.array([t[1] for i, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

                TE_start_inds[tsk, te] = start_times
                TE_go_inds[tsk, te] = []
                #Categorize trials: 
                #Skip first trial: 
                for trial_ind, ix in enumerate(start_ind):
                                    #Get Trial outcome
                    if ix+7 >= msg.shape[0]: #Very end of block
                        print 'before: ', TE_start_inds[tsk, te].shape
                        new_start_times = np.delete(TE_start_inds[tsk, te],[trial_ind])

                        TE_start_inds[tsk, te] = new_start_times
                        print 'must delete index! '
                        print 'after: ', TE_start_inds[tsk, te].shape

                    else:
                        #Get Row of table: 
                        trial = table.row

                        trial['task_entry'] = te
                        trial['trial_type'] = tsk
                        trial['start_time'] = start_times[trial_ind]

                        kinrow = kin_table.row
                        kinrow['task_entry'] = te
                        kinrow['start_time'] = start_times[trial_ind]

                        if ix < len(msg)+7:
                            if tsk in ['S1','S4','M1','M4']:
                                trial, kinrow, TE_go_inds = mc_metrics.get_CO_metrics([trial_ind, ix], msg, msg_time, 
                                    trial, kinrow, TE_go_inds,rew_times, hdf,tsk, te)
                            
                            elif tsk in ['manualcontrol_memory']:
                                trial, kinrow, TE_go_inds = mc_metrics.get_mc_mem_metrics([trial_ind, ix], msg, msg_time, 
                                    trial, kinrow, TE_go_inds, rew_times, hdf,tsk, te)

                        trial.append()
                        kinrow.append()
                kin_table.flush()       
                table.flush()
            h5file.close()
        sio.savemat(self.task_entry_dict_time_inds_fname, TE_start_inds)
        sio.savemat(self.task_entry_dict_go_times_fname, TE_go_inds)
        
        self.task_entry_dict_time_inds = TE_start_inds
        self.go_times = TE_go_inds

if __name__ == "__main__":
    # d = dict(behav_file_name='new_cart_behav',\
    #       neural_file_name = 'new_cart_neural',\
    #       task_entry_dict_fname='task_entries_reaching_dec14.mat',\
    #       task_entry_dict_go = 'task_entries_reaching_go_dec14.mat',\
    #       t_range=[1, 2.5],\
    #       )

    # #d['task_entry_dict_fname'] = 'task_entries_trunc_dec14.mat'

    # mcd = manual_control_data(**d)
    

    # #Get behav: 
    # mcd.get_behavior()


    # kw = dict(t_range=[1,2.5],use_go_file=True,small_f_steps=True)

    # jobs = []
    # for tsk in mcd.tasks:
    # #for tsk in ['S4']:
    #   print 'tsk: ', tsk
    #   p = multiprocessing.Process(target=mcd.make_neural,args=([tsk],),kwargs=kw)
    #   jobs.append(p)
    #   p.start()
    #   #mcd.make_neural(tsk)

    d = dict(behav_file_name='pap_rev_new_cart_behav',\
            neural_file_name = 'pap_rev_new_cart_welch_neural',\
            task_entry_dict_fname='task_entries_reaching_dec14.mat',\
            task_entry_dict_go = 'task_entries_reaching_go_dec14.mat',\
            t_range=[1, 2.5],\
            spec_method='Welch'
            )

    #d['task_entry_dict_fname'] = 'task_entries_trunc_dec14.mat'

    mcd = manual_control_data(**d)

    #Get behav: 
    mcd.get_behavior()


    kw = dict(t_range=[1,2.5],use_go_file=True,small_f_steps=True)

    jobs = []
    for tsk in mcd.tasks:
    #for tsk in ['S4']:
        print 'tsk: ', tsk
        p = multiprocessing.Process(target=mcd.make_neural,args=([tsk],),kwargs=kw)
        jobs.append(p)
        p.start()
        #mcd.make_neural(tsk)

