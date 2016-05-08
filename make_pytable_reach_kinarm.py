from make_pytable_reach_bmi3d import Behav_Hand_Reach, Neural_Hand_Reach, Kin_Traces
import parse
import spectral_metrics as sm
import simple_spec as ss
import psycho_metrics as pm
from targ_dir import target_direction_array
import gc
import multiprocessing
import tables
import scipy.io as sio
import numpy as np
import fcns
import datetime
import tables

class Behav_Hand_Reach_kinarm(Behav_Hand_Reach):
    task_entry = tables.StringCol(7)

class Neural_Hand_Reach_kinarm(Neural_Hand_Reach):
    task_entry = tables.StringCol(7)
    power_sig_kinarm = tables.Float32Col(shape =(296,26)) #time x freq x channel
    #long_neural_sig = tables.Float32Col(shape = (5500, 1))
    long_power_sig_kinarm = tables.Float32Col(shape = (409, 103))

class Neural_Hand_Reach_kinarm_neur_only(tables.IsDescription):
    long_neural_sig = tables.Float32Col(shape = (5500, 1))
    neural_sig = tables.Float32Col(shape = (3500, 1))   
    trial_type = tables.StringCol(2)
    task_entry = tables.IntCol()
    start_time = tables.IntCol()

class Kin_Traces_kinarm(Kin_Traces):
    task_entry = tables.StringCol(7)
    kin_sig_kinarm = tables.Float64Col(shape=(3000, ))

class kinarm_manual_control_data(object):
    def __init__(self, *args, **kwargs):
        self.dates = kwargs['dates'] #dictionary of dates
        self.blocks = kwargs['blocks'] #dictionary of blocks
        self.direct = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'
        self.anim = kwargs.pop('anim', 'seba')
        print self.anim

        if 'tasks' not in kwargs.keys():
            self.tasks = ['S1','S4','M1','M4']
        else:
            self.tasks = kwargs['tasks']

        self.channels = kwargs['channels']

        #check files and kin files are made and all channels are present: 
        for tsk in self.tasks:
            e = parse.epoch(dates = self.dates[tsk], blocks=self.blocks[tsk], chans = self.channels, 
                kin_only = False, anim=self.anim)
            ek = parse.epoch(dates = self.dates[tsk], blocks=self.blocks[tsk], kin_only = True,
                anim=self.anim)
            e.mat_file_check()
            ek.mat_file_check()

        self.behav_fname = kwargs['behav_fname']
        self.neur_fname = kwargs['neur_fname']
        self.task_entry_dict_time_inds_fname = kwargs['task_entry_dict_time_inds_fname']
        self.task_entry_dict_go_times_fname = kwargs['task_entry_dict_go']
        self.spec_method =kwargs.pop('spec_method', 'MTM')

        tdy = datetime.date.today()
        self.tdy_str = tdy.isoformat()
        self.moving_window = kwargs.pop('moving_window', [.251, .011])
        self.neural_sig_only = kwargs.pop('neural_sig_only', False)

    def make_neural(self, task_name, **kwargs):
        if 't_range' in kwargs.keys():
            before_go = kwargs['t_range'][0]
            after_go = kwargs['t_range'][1]
        else:
            before_go = 1
            after_go = 2.5

        if before_go+after_go > 3.5:
            self.long_trials = True
            self.bp_filt = [.10, 55]
        else:
            self.long_trials = False
            self.bp_filt = [10, 55]

        neural_sig_only = self.neural_sig_only

        if neural_sig_only:
            self.neur_fname = self.neur_fname + 'neur_sig_only_'

        if 'use_go_file' in kwargs.keys():
            try:
                task_entry_dict_time_inds = self.go_times
                str_flag = False

            except:
                task_entry_dict_time_inds = sio.loadmat(self.task_entry_dict_go_times_fname)
                str_flag = True
        else:
            if hasattr(self, 'task_entry_dict_time_inds'):
                str_flag = False
                task_entry_dict_time_inds = self.task_entry_dict_time_inds
            else:
                task_entry_dict_time_inds = sio.loadmat(self.task_entry_dict_time_inds_fname)
                str_flag = True

        start_time_dict = sio.loadmat(self.task_entry_dict_time_inds_fname)

        for tsk in task_name:
            moving_window = self.moving_window

            dats = self.dates[tsk]
            blks = self.blocks[tsk]

            tdy = datetime.date.today()
            tdy_str = tdy.isoformat()

            h5file = tables.openFile(self.tdy_str + self.neur_fname + tsk + '.h5', mode="w", title='Seba, neural')

            if neural_sig_only:
                table = h5file.createTable("/", 'neural', Neural_Hand_Reach_kinarm_neur_only, "Neural Table")
            else:
                table = h5file.createTable("/", 'neural', Neural_Hand_Reach_kinarm, "Neural Table")
            
            for i_d, d in enumerate(dats):
                for i_b, b in enumerate(blks[i_d][0]):
                    if str_flag: 
                        start_times = np.squeeze(np.array(task_entry_dict_time_inds[str((tsk, d, b))]))
                    else:
                        start_times = np.squeeze(np.array(task_entry_dict_time_inds[tsk, d, b]))
                    
                    mat = sio.loadmat(self.direct+'seba'+d+b+'.mat')
                    signal = dict()
                    smtm = dict()
                    
                    for c,ch in enumerate(self.channels):
                        ch_key = 'AD'+str(ch)
                        print 'I . AM', self.anim, '-ador'
                        signal[ch_key], _ = sm.get_sig([d],[b],start_times,[len(start_times)],before_go=before_go, after_go=after_go, 
                            channel=ch_key, anim=self.anim)

                        if neural_sig_only:
                            print 'skipping spec stuff'
                            f = np.arange(1, 100)
                        
                        else:
                            if self.spec_method == 'MTM':
                                Smtm, f, t = ss.MTM_specgram(signal[ch_key].T,movingwin=moving_window)
                            elif self.spec_method == 'Welch':
                                print 'Using welch!'
                                Smtm, f, t = ss.Welch_specgram(signal[ch_key].T, movingwin=moving_window, bp_filt=self.bp_filt)
                            smtm[ch_key] = Smtm

                    f_trim = f[f< 100]
                    f_trim_ix = f<100

                    for i_t in range(len(start_times)):
                        trl = table.row
                        if neural_sig_only:
                            sgg = np.zeros((signal[ch_key].shape[1], len(self.channels)))
                            for ic, c in enumerate(self.channels):
                                sgg[:,ic] = signal['AD'+str(c)][i_t,:]
                            
                            if self.long_trials:
                                trl['long_neural_sig'] = sgg
                            
                            else:
                                trl['neural_sig'] = sgg
                        else:
                            pxx = np.zeros((Smtm.shape[1], Smtm.shape[2], len(self.channels)))
                            for ic, c in enumerate(self.channels):
                                pxx[:,:,ic] = smtm['AD'+str(c)][i_t,:,:]
                            if self.long_trials:
                                #trl['long_neural_sig'] = sgg
                                trl['long_power_sig_kinarm'] = pxx[:, f_trim_ix, 0]
                            else:
                                #trl['neural_sig'] = sgg
                                trl['power_sig_kinarm'] = pxx[:, f_trim_ix, 0]

                        trl['trial_type'] = tsk[0]
                        trl['task_entry'] = d+b
                        trl['start_time'] = np.squeeze(start_time_dict[str((tsk, d, b))])[i_t]
                        trl.append()
                    table.flush()
            if neural_sig_only:
                add_cols = h5file.createGroup(h5file.root, "columns", "t_range")
                h5file.createArray(add_cols, 't_range', np.array([before_go, after_go]))
            else:
                add_cols = h5file.createGroup(h5file.root, "columns", "Channels, Freq, Bins")
                h5file.createArray(add_cols, 'channels', np.array(self.channels))
                h5file.createArray(add_cols, 'freq', f)
                h5file.createArray(add_cols, 'bins', t)
            h5file.close()

    def get_behavior(self):
        TD_array = target_direction_array()
        task_start_times = dict()
        task_go_times = dict()
        for i, tsk in enumerate(self.tasks):
            h5file = tables.openFile(self.tdy_str + self.behav_fname + tsk + '.h5', mode="w", title='Seba, behavior')
            table = h5file.createTable("/", 'behav', Behav_Hand_Reach_kinarm, "Behavior Table")
            kin_table = h5file.createTable("/",'kin',Kin_Traces_kinarm,"Kin Table")

            for d, day in enumerate(self.dates[tsk]):
                for b, bl in enumerate(self.blocks[tsk][d][0]):
                    mat_fl = sio.loadmat(self.direct+'seba'+day+bl+'.mat')
                    strobed = mat_fl['Strobed'][:,1]
                    strobed_tm = mat_fl['Strobed'][:,0]

                    #trial starts: 
                    start_ind = np.nonzero(strobed==2)[0]
                    start_times = strobed_tm[start_ind]
                    task_start_times[tsk, day, bl] = start_times
                    task_go_times[tsk,day,bl] = []

                    for trial_ind, ix in enumerate(start_ind):
                        trial = table.row
                        kinrow = kin_table.row

                        trial['trial_type'] = tsk

                        if ix+7 >= strobed.shape[0]: #Very end of block
                            #print 'before: ', task_start_times[tsk, day,bl]
                            new_start_times = np.delete(task_start_times[tsk, day,bl],[trial_ind])
                            task_start_times[tsk, day,bl] = new_start_times
                            #print 'must delete index! '
                            #print 'after: ', task_start_times[tsk, day,bl]

                        else:
                            #Get trial outcome: 
                            if strobed[ix+2]==200:
                                trial_outcome = 'center_timeout'
                                task_go_times[tsk,day,bl].append(strobed_tm[ix])
                            elif strobed[ix+3]==4:
                                trial_outcome = 'center_holderr'
                                task_go_times[tsk,day,bl].append(strobed_tm[ix])
                            elif (strobed[ix+4]==4) or (strobed[ix+5]==12):
                                trial_outcome = 'periph_timeout'
                                task_go_times[tsk,day,bl].append(strobed_tm[ix+3])
                            elif strobed[ix+6]==8:
                                trial_outcome = 'periph_holderr'
                                task_go_times[tsk,day,bl].append(strobed_tm[ix+3])
                            elif strobed[ix+7]==9:
                                trial_outcome = 'reward_trial__'
                                task_go_times[tsk,day,bl].append(strobed_tm[ix+3])
                            else:
                                print 'error: unrecognized trial type, msg index: ', ix, day, bl

                            trial['trial_outcome'] = trial_outcome

                            #Get reach time: 
                            if trial_outcome in ['periph_holderr', 'reward_trial__']:

                                #MC label: 
                                mc = strobed[ix+1]
                                go_cue_time = strobed_tm[ix+3]
                                
                                #Cursor trajectory: 
                                kin_sig, targ_dir = sm.get_sig([day], [[bl]], np.array([go_cue_time]), [1], signal_type='shenoy_jt_vel',mc_lab = np.array([mc]), prep=True)
                                kin_feat = pm.get_kin_sig_shenoy(kin_sig) #Kin feats #Take index #2: 
                                trial['rxn_time'] = kin_feat[0, 2]
                                trial['reach_time'] = strobed_tm[ix+5] - strobed_tm[ix+3]
                                trial['target_loc'] = targ_dir[0,:]
                                kinrow['mc_vect'] = targ_dir
                                kinrow['kin_sig_kinarm'] = kin_sig
                                kinrow['kin_feat'] = kin_feat[0,:]

                            else:
                                trial['rxn_time'] = -1
                                trial['reach_time'] = -1
                                trial['target_loc'] = TD_array.targ_dir[strobed[ix+1]]

                            if trial_outcome not in ['center_timeout', 'center_holderr']:
                                trial['hold_time'] = strobed_tm[ix+3] - strobed_tm[ix+2]
                            else:
                                trial['hold_time'] = -1

                            #Reach Error
                            #From onset of movement
                            # kin_sig_curs, targ_dir = sm.get_sig([day], [[bl]], np.array([go_cue_time]), [1], signal_type='endpt',mc_lab = np.array([mc]), prep=True)
                            # curs_trunc = kin_sig_curs[0,1000:,:]          

                            #Reward Rate: 
                            rew_ind = np.nonzero(strobed==9)[0]
                            rew_times = strobed_tm[rew_ind]

                            start_time = strobed_tm[ix]
                            end_session_time = strobed_tm[-1]

                            if start_time < 0.5*60:
                                rews = np.nonzero(rew_times<start_time+(0.5*60))[0]
                                rew_rate = len(rews)/float(0.5+ start_time/(60.))

                            elif start_time + 0.5*60 > end_session_time:
                                rews = np.nonzero(rew_times>start_time-(0.5*60.))[0]
                                rew_rate = len(rews)/float(0.5+ (end_session_time - start_time)/(60.))

                            else:
                                rix, rval = fcns.get_in_range(rew_times, [start_time - (0.5*60.), start_time + (0.5*60.)])
                                rew_rate = len(rix)
                            trial['rew_rate'] = rew_rate
                            trial['start_time'] = start_time
                            trial['task_entry'] = day+bl

                            kinrow['start_time'] = start_time
                            kinrow['task_entry'] = day+bl

                            if trial_outcome not in ['center_holderr', 'center_timeout']:
                                trial['go_time'] = strobed_tm[ix+3]
                            else:
                                trial['go_time'] = -1

                            trial.append()
                            kinrow.append()

                    kin_table.flush()
                    table.flush()
                    gc.collect()
            h5file.close()
        sio.savemat(self.task_entry_dict_time_inds_fname, task_start_times)
        sio.savemat(self.task_entry_dict_go_times_fname, task_go_times)

        self.task_entry_dict_time_inds = task_start_times
        self.go_times = task_go_times
if __name__ == "__main__":
    # d = dict(behav_fname='new2_seba_behav',\
    #         neur_fname = 'new2_seba_neural',\
    #         dates = dict(S1=['102414','102514'],\
    #                     S4=['102414','102514','110414'],\
    #                     M1=['102414','102514','110114','110314'],\
    #                     M4=['102514', '110114','110314','110414']),\
    #         blocks = dict(S1=[['ac'],['a']],\
    #                     S4=[['b'],['b'],['b']],\
    #                     M1=[['d'],['ce'],['d'],['a']],\
    #                     M4=[['d'], ['a'], ['i'], ['a']]), \
    #         channels=[70,72, 76, 140,162], \
    #         #channels=[70], \
    #         task_entry_dict_time_inds_fname='tsk_tms_kinarm_reaching_nov14.mat',\
    #         task_entry_dict_go = 'tsk_tms_kinarm_reaching_go_nov14.mat',\
    #         )
    #     # d = dict(behav_fname='new_seba_behav',\
    #     #         neur_fname = 'new_seba_neural',\
    #     #         dates = dict(S1=['102414']),
    #     #         blocks = dict(S1=[['ac'],['a']]),
    #     #         #channels=[70],72,76,140,141,142,162,163], \
    #     #         channels=[70], \
    #     #         task_entry_dict_time_inds_fname='tsk_tms_kinarm_reaching_nov14.mat'
    #     #         )
    # mcd = kinarm_manual_control_data(**d)

    # #Get behav: 
    # mcd.get_behavior()

    # jobs = []
    # kw = dict(t_range=[1,2.5],use_go_file=True)
    # #mcd.tasks=['S1']
    # for tsk in mcd.tasks:
    #     print 'tsk: ', tsk
    #     p = multiprocessing.Process(target=mcd.make_neural,args=([tsk],),kwargs=kw)
    #     jobs.append(p)
    #     p.start()

######### REAL D: ###########
    d = dict(behav_fname='pap_rev_seba_behav',\
            neur_fname = 'pap_rev_welch_seba_neural',\
            dates = dict(S1=['102414','102514'],\
                        M1=['102414','102514','110114','110314']),\
            blocks = dict(S1=[['ac'],['a']],\
                        M1=[['d'],['ce'],['d'],['a']]),\
            #channels=[70,72, 76, 140,162], \
            channels=[70], \
            task_entry_dict_time_inds_fname='tsk_tms_kinarm_reaching_nov14.mat',\
            task_entry_dict_go = 'tsk_tms_kinarm_reaching_go_nov14.mat',\
            tasks=['S1','M1'],
            spec_method='Welch')

# ######### Trunc D: ###########
#     d = dict(behav_fname='pap_rev_seba_behav',\
#             neur_fname = 'pap_rev_seba_neural',\
#             dates = dict(S1=['102414']),\
#             blocks = dict(S1=[['a']]),\
#             #channels=[70,72, 76, 140,162], \
#             channels=[70], \
#             task_entry_dict_time_inds_fname='tsk_tms_kinarm_reaching_nov14.mat',\
#             task_entry_dict_go = 'tsk_tms_kinarm_reaching_go_nov14.mat',\
#             tasks=['S1'], \
#             spec_method='Welch')

    mcd = kinarm_manual_control_data(**d)

    #Get behav: 
    mcd.get_behavior()

    jobs = []
    kw = dict(t_range=[1,2.5],use_go_file=True, spec_method='Welch')

    mcd.make_neural(['M1'], **kw)
    # #mcd.tasks=['S1']
    # for tsk in mcd.tasks:
    #     print 'tsk: ', tsk
    #     p = multiprocessing.Process(target=mcd.make_neural,args=([tsk],),kwargs=kw)
    #     jobs.append(p)
    #     p.start()



