from make_pytable_lfpmod_bmi3d import Behav_LFPMod_Trial, Behav_Hand_Reach_LFP
from make_pytable_reach_kinarm import Kin_Traces_kinarm, kinarm_manual_control_data
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

class Behav_LFPMod_Trial_kin(Behav_LFPMod_Trial):

    task_entry = tables.StringCol(7)

class Behav_Hand_Reach_LFP_kin(Behav_Hand_Reach_LFP):
    task_entry = tables.StringCol(7)

class Kin_Traces_kinarm_lfp(Kin_Traces_kinarm):
    kin_sig = tables.Float64Col(shape=(3000,))

class kinarm_lfpmod_data(kinarm_manual_control_data):
    def __init__(self, *args, **kwargs):
        super(kinarm_lfpmod_data, self).__init__(*args,**kwargs)
        self.task_entry_dict_time_inds_fname2 = 's_idx_'+self.task_entry_dict_time_inds_fname

    def get_behavior(self):
        TD_array = target_direction_array()
        task_go_times = dict()
        task_start_times = dict()
        task_entry_dict_time_inds = dict()
        task_start_indices = dict()

        for i, tsk in enumerate(self.tasks):
            try:
                h5file = tables.openFile(self.tdy_str + self.behav_fname + tsk + '.h5', mode="w", title='Seba, behavior')
            except:
                h5file = tables.openFile('x'+self.tdy_str + self.behav_fname + tsk + '.h5', mode="w", title='Seba, behavior')

            table = h5file.createTable("/", 'behav', Behav_Hand_Reach_LFP_kin, "Behavior Table")
            kin_table = h5file.createTable("/",'kin',Kin_Traces_kinarm_lfp,"Kin Table")

            for d, day in enumerate(self.dates[tsk]):
                for b, bl in enumerate(self.blocks[tsk][d][0]):
                    task_entry_dict_time_inds[tsk, day, bl] = []
                    task_go_times[tsk, day, bl] = []
                    task_start_times[tsk, day, bl] = []

                    mat_fl = sio.loadmat(self.direct+'seba'+day+bl+'.mat')
                    strobed = mat_fl['Strobed'][:,1]
                    strobed_tm = mat_fl['Strobed'][:,0]

                    #trial rewards: 
                    rew_indices = np.nonzero(strobed==9)[0]
                    start_indices = np.nonzero(strobed==2)[0]



                    for trial_ind, ix in enumerate(rew_indices):
                        trial = table.row
                        kinrow = kin_table.row

                        trial['trial_type'] = tsk

                        if ix+7 >= strobed.shape[0]: #Very end of block
                            start_indices = np.delete(start_indices, [trial_ind])
                            rew_indices = np.delete(rew_indices, [trial_ind])
                            print 'before: ', task_go_times[tsk, day,bl]
                            #new_start_times = np.delete(task_start_times[tsk, day,bl],[trial_ind])
                            #task_start_times[tsk, day,bl] = new_start_times
                            #print 'must delete index! '
                            #print 'after: ', task_start_times[tsk, day,bl]

                        else:

                            #Get new part: 
                            # [ 2, 15, 66, 84, 28, 84, ..., 28, 5, ,6, 7, 9, 30]

                            # Looking for PHE or REWARD: 
                            trial['trial_outcome'] = 'reward'

                            # Get start index / start time: 
                            start_idx = np.max(start_indices[start_indices<ix])
                            if int(strobed[start_idx+2]) in np.arange(60,80):
                                mc = strobed[start_idx+2]
                            elif int(strobed[start_idx+2]) == 15:
                                if int(strobed[start_idx+3]) in np.arange(60,80):
                                    mc = strobed[start_idx+3]
                            else:
                                print ' helllllllp!'

                            if tsk == 'early':
                                mc = strobed[start_idx+2] #MC target'
                            elif tsk == 'med':
                                mc = strobed[start_idx+1] #

                            lfp = strobed[start_idx+3]

                            tmp = strobed[start_idx:ix]
                            go_cue_sub_idx = np.nonzero(tmp==5)[0]
                            go_cue_idx = start_idx+go_cue_sub_idx

                            task_go_times[tsk, day, bl].extend([strobed_tm[go_cue_idx]])
                            task_start_times[tsk, day, bl].extend([strobed_tm[strobed_tm[start_idx]]])

                            if strobed[go_cue_idx] != 5:
                                print 'helpppppP!'

                            go_cue_time = strobed_tm[go_cue_idx]
                            
                            #Cursor trajectory: 
                            kw = dict(target_coding='standard')

                            kin_sig, targ_dir = sm.get_sig([day], [[bl]], np.array([go_cue_time]), [1], signal_type='shenoy_jt_vel',mc_lab = np.array([mc]), prep=True,**kw)
                            kin_feat = pm.get_kin_sig_shenoy(kin_sig) #Kin feats #Take index #2: 
                            
                            trial['target_loc_mc'] = targ_dir[0,:]
                            trial['target_loc_lfp'] = lfp
                            trial['rxn_time'] = kin_feat[0, 2]
                            trial['reach_time'] = strobed_tm[go_cue_idx+2] - strobed_tm[go_cue_idx]
                            trial['target_loc'] = targ_dir[0,:]
                            trial['go_time'] = strobed_tm[go_cue_idx]

                            kinrow['mc_vect'] = targ_dir
                            kinrow['kin_sig'] = kin_sig
                            kinrow['kin_feat'] = kin_feat[0,:]
                            
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

                            trial['go_time'] = strobed_tm[ix+3]
                            
                            trial.append()
                            kinrow.append()
                    task_start_indices[tsk, day, bl] = start_indices
                    kin_table.flush()
                    table.flush()
                    gc.collect()
            h5file.close()
        sio.savemat(self.task_entry_dict_time_inds_fname, task_start_times)
        sio.savemat(self.task_entry_dict_go_times_fname, task_go_times)
        sio.savemat(self.task_entry_dict_time_inds_fname2, task_start_indices)

        self.task_start_times = task_start_times
        self.go_times = task_go_times
        self.start_idx = task_start_indices 

    def get_lfpmod_behavior(self):
        for k, tsk in enumerate(self.tasks):
            h5file = tables.openFile(self.tdy_str + self.behav_fname +tsk+'.h5', mode="w", title='Seba, lfp_behavior')
            table_trial = h5file.createTable("/", 'trl_bhv', Behav_LFPMod_Trial_kin, "Trial Behavior Table")

            for d, day in enumerate(self.dates[tsk]):
                for b, bl in enumerate(self.blocks[tsk][d][0]):
                    mat_fl = sio.loadmat(self.direct+'seba'+day+bl+'.mat')
                    strobed = mat_fl['Strobed'][:,1]
                    strobed_tm = mat_fl['Strobed'][:,0]


                    try:
                        go_times = self.go_times[tsk,day,bl]
                    
                    except:
                        go = sio.loadmat(self.task_entry_dict_go_times_fname)
                        go_times = np.squeeze(go[str((tsk,day,bl))])

                    try: 
                        print self.task_start_times[tsk,day,bl][0]
                        use_str = False
                    except:
                        self.task_start_times = sio.loadmat(self.task_entry_dict_time_inds_fname)
                        self.start_idx = sio.loadmat(self.task_entry_dict_time_inds_fname2)
                        use_str = True

                    go_indices = np.nonzero(strobed==5)[0]

                    for i, g in enumerate(go_times):
                        trial = table_trial.row
                        trial['task_entry'] = day+bl #for syncing later

                        if use_str:
                            trial_start_time = self.task_start_times[str((tsk,day,bl))][i]
                            start_id = self.start_idx[str((tsk,day,bl))][i]
                        else:
                            trial_start_time = self.task_start_times[tsk,day,bl][i]
                            start_id = self.start_idx[tsk,day,bl][i]

                        trial['start_time'] = trial_start_time
                        trial['lfp_targ_loc'] = strobed[start_id+3]
                        trial['reach_time'] = g - trial_start_time

                        go_ix = np.min(go_indices>start_id)
                        trial['hold_time'] = strobed[go_ix] - strobed[go_ix - 1]
                        trial.append()
                table_trial.flush()
        h5file.close()

if __name__ == "__main__":
    import sys
    d = dict(behav_fname='pap_rev_seba_behav',\
        neur_fname = 'pap_rev_seba_welch_neural',\
        #channels=[70,72, 76, 140,162], \
        channels=[70], \
        tasks = ['lfp_mod_mc_reach_out'],\
        spec_method='Welch',\
        moving_window = [.251, .011], \
        t_range = [1.5, 2]
        )

    arg_ind = int(sys.argv[1])
    if arg_ind ==1:
        d['dates'] = dict(lfp_mod_mc_reach_out = ['082714','082814','082914'])
        d['blocks'] = dict(lfp_mod_mc_reach_out= [['bcdefg'], ['bcdefghij'], ['bcdefgh']])
        d['task_entry_dict_time_inds_fname'] = 'seba_prep_t1.mat'
        d['task_entry_dict_go'] = 'seba_prep_t1_GO.mat'
        d['behav_fname']='pap_rev_seba_behav_t1'
        d['neur_fname'] = 'pap_rev_seba_neur_t1'

    elif arg_ind == 11:
        d['dates'] = dict(lfp_mod_mc_reach_out = ['090314','090414','090514','110314'])
        d['blocks'] = dict(lfp_mod_mc_reach_out= [['efghij'],['efg',],['bcdefg'],['bc']])
        d['task_entry_dict_time_inds_fname'] = 'seba_prep_t1_power_control.mat'
        d['task_entry_dict_go'] = 'seba_prep_t1_GO_power_control.mat'        
        d['behav_fname']='pap_rev_seba_behav_t1_power_control'
        d['neur_fname'] = 'pap_rev_seba_neur_t1_power_control'

    if arg_ind == 12:
        d['dates'] = dict(lfp_mod_mc_reach_out = ['082714','082814'])#,'082914'])
        d['blocks'] = dict(lfp_mod_mc_reach_out= [['bc'], ['bc']]) #defg'], ['bcdefghij'], ['bcdefgh']])
        d['task_entry_dict_time_inds_fname'] = 'seba_prep_t1.mat'
        d['task_entry_dict_go'] = 'seba_prep_t1_GO.mat'
        d['behav_fname']='pap_rev_seba_behav_t1_large_window'
        d['neur_fname'] = 'pap_rev_seba_neur_t1_large_window'
        d['moving_window'] = [1.001, .011]
        d['t_range'] = [2.5, 3]

    elif arg_ind ==3:
        d['dates'] = dict(lfp_mod_mc_reach_out = ['111414','111514','111714','111814'])
        d['blocks'] = dict(lfp_mod_mc_reach_out= [['bcd'], ['bcdefghi'],['cdeghijk'],['abcdefgh']])
        d['task_entry_dict_time_inds_fname'] = 'seba_prep_t3.mat'
        d['task_entry_dict_go'] = 'seba_prep_t3_GO.mat'
        d['behav_fname']='pap_rev_seba_behav_t3'
        d['neur_fname'] = 'pap_rev_seba_neur_t3'

    elif arg_ind == 31:
        d['dates'] = dict(lfp_mod_mc_reach_out = ['110314','110514','110614'])
        d['blocks'] = dict(blocks = [['defgh'], ['bcdefghijk'], ['bcdefghijk']])
        d['task_entry_dict_time_inds_fname'] = 'seba_prep_t3_power_control.mat'
        d['task_entry_dict_go'] = 'seba_prep_t3_GO_power_control.mat' 
        d['behav_fname']='pap_rev_seba_behav_t3_power_control'
        d['neur_fname'] = 'pap_rev_seba_neur_t3_power_control'   

    elif arg_ind == 32:
        d['dates'] = dict(lfp_mod_mc_reach_out = ['111414','111514','111714','111814'])
        d['blocks'] = dict(lfp_mod_mc_reach_out= [['bcd'], ['bcdefghi'],['cdeghijk'],['abcdefgh']])
        d['task_entry_dict_time_inds_fname'] = 'seba_prep_t3_large_window.mat'
        d['task_entry_dict_go'] = 'seba_prep_t3_GO_large_window.mat'
        d['behav_fname']='pap_rev_seba_behav_t3_large_window'
        d['neur_fname'] = 'pap_rev_seba_neur_t3_large_window'
        d['moving_window'] = [1.001, .011]  
        d['t_range'] = [2.5, 3]

    mcd = kinarm_lfpmod_data(**d)
    
    #Get behav: 
    mcd.get_behavior()
    
    kw = dict(t_range=d['t_range'],use_go_file=True, spec_method='Welch')
    mcd.moving_window = d['moving_window']

    for tsk in mcd.tasks:
        print 'tsk: ', tsk
        mcd.make_neural([tsk], **kw)

