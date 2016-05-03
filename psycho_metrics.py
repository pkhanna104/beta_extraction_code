import numpy as np
import scipy.io as sio
#import parse
import scipy
import matplotlib.pyplot as plt
import spectral_metrics as sm
import sys
from sklearn.preprocessing import normalize
import sav_gol_filt as sg_filt
import gc


# #Sessions with 450 ms LFP hold time: 
# s2_dates = ['042914','043014','050214'] 
# s2_blocks=[['ij'],['ab'],['abcdefghij']]

# s4_dates = ['042914','043014',] 
# s4_blocks=[['abcdefgh'],['cdefghi']]

# fin_dates = ['050314','050414','050514','050614','050714','050814'] 
# fin_blocks=[['efg'],['abdefg'],['abcdefghij'],['abcdeghijkl'],['abcdefghij'],['abcdefghi']]

# #Make mat files: 
# fin_dates = ['050314','050414','050514','050614','050714','050814'] 
# fin_blocks=[['efg'],['abdefg'],['abcdefghij'],['abcdefghijkl'],['abcdefghij'],['abcdefghi']]

# cue_on_dates = ['050914','051314','051414','051514','051614']
# cue_on_blocks = [['efgh'], ['abcdefghij'],['abcdefghijk'],['abcdefghi'],['abcdefghi']]

# cue_med_dates = ['051814', '051914','052014']
# cue_med_blocks = [['hij'],['abcdefghij'],['abcdefghij']]

# sim_lfp_dates = ['052214','052314','052814']
# sim_lfp_blocks= [['abcdefghij'], ['abcdefgh'], ['abcdefghijk']]

# e = parse.epoch(5,dates=cue_on_dates, blocks=cue_on_blocks)
# e.mat_file_check()

class codes():
    def __init__(self,include_mctarg_5=True,LFP_only=False,new_pre=False,sdh_pre=False):
        if new_pre:
            self.pre = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/manual_control_redo/'
        else:
            self.pre = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'

        if sdh_pre:
            self.pre = '/home/preeya/grom_mats/'
            print sdh_pre, 'SDH PRE '

        self.kin_param_path = '/Users/preeyakhanna/Dropbox/Carmena_Lab/lfp_multitask/analysis/'
        self.trial_start = 2
        self.mc_go = 5
        self.move_onset = 6
        self.enter_periph = 7
        self.rew = 9

        self.lfp_enter_periph = 28
        self.lfp_targ = [84,85,86,87]
        self.lfp_only_go = 15
        self.lfp_only_rew = 30

        if include_mctarg_5:
            self.mc_targ = [64,66,68,70]
        
        else:
            self.mc_targ = [64,66,70]

        self.LFP_only = LFP_only

        self.cmap = ['lightseagreen','dodgerblue','gold','orangered','k']

def RT_files(dates,blocks,key,fname,sequential=True,task='multi',key_save=None, **kwargs):
    sequ = dict()

    if 'only_basics' in kwargs.keys():
        abs_time_go, n = get_MC_RTs(dates,blocks,key,sequential=True,task=task,**kwargs)
        trial_type = 0

    if 'only_basics_plus_MC' in kwargs.keys():
        print 
        abs_time_go, n, MC_label = get_MC_RTs(dates,blocks,key,sequential=True,task=task,**kwargs)
        trial_type = 0
        sequ['MC_label'] = MC_label
    else:

        if task=='LFP_only': #NOT IMPLEMENTED ELSEWHERE, fix
            LFP_RT, LFP_label, n, abs_time_go = get_LFP_RT(dates,blocks,key,sequential=True,task=task,**kwargs) 
            sequ['LFP_RT'] = LFP_RT
            sequ['LFP_label'] = LFP_label
        
        elif scipy.logical_or(task=='MC',task=='mc'):
            MC1_RT, MC2_RT, MC_label, LFP_label, abs_time_go, n, trial_type = get_MC_RTs(dates,blocks,key,
                sequential=True,task=task,**kwargs)
            sequ['MC1_RT'] = MC1_RT
            sequ['MC2_RT'] = MC2_RT
            sequ['MC_label'] = MC_label

        elif task == 'targ_jump':
            MC1_RT, MC2_RT, MC_label1, MC_label2, abs_time_go, n, trial_type = get_MC_RTs(dates,blocks,key,
                sequential=True,task=task,**kwargs)
            #Here, MC1 = 5--> 6, MC2 = t2 -->7
            #MC_label1 = MC targ 1
            #MC_label2 = MC targ 2
            sequ['MC1_RT'] = MC1_RT
            sequ['MC2_RT'] = MC2_RT
            sequ['MC_label1'] = MC_label1
            sequ['MC_label2'] = MC_label2

        elif task == 'phaseBMI':
            phase_RT, phase_label, n, abs_time_go = get_LFP_RT(dates,blocks,key,sequential=True, 
                task=task,**kwargs)
            sequ['phase_RT'] = phase_RT
            sequ['phase_label'] = phase_label
            trial_type = np.ones((np.sum(n),))+8    

        else:
            LFP_RT, LFP_label, n = get_LFP_RT(dates,blocks,key,sequential=True,task=task,**kwargs)
            MC1_RT, MC2_RT, MC_label, LFP_label, abs_time_go, n, trial_type = get_MC_RTs(dates,blocks,key,
                sequential=True,task=task,**kwargs)
            
            sequ['MC1_RT'] = MC1_RT
            sequ['MC2_RT'] = MC2_RT
            sequ['MC_label'] = MC_label
            sequ['LFP_RT'] = LFP_RT
            sequ['LFP_label'] = LFP_label
    
    sequ['abs_time_go'] = abs_time_go
    sequ['n'] = n
    sequ['trial_type'] = trial_type
    if fname is not False:
        if key_save is None:
            sio.savemat(key.pre+fname,sequ)
        else:
            sio.savemat(key_save.pre+fname,sequ)
    else:
        return sequ


def get_LFP_RT(dates,blocks,key,sequential=False,task='multi',**kwargs):
    '''
    Assumes code structure is 2,15, cotarg, lfptarg, 28, 5,6,7,9,30
    This take RT as (28) - lfptarg 

    sequential = true returns a 1d array with RT and targets 
    in the order that they occurred
    '''
    if key.LFP_only:
        rew_key = key.lfp_only_rew
    else:
        rew_key = key.rew

    if ('use_kin_strobed' in kwargs.keys() and kwargs['use_kin_strobed']):
        fname_ext='_kin'
        backup_ext = ''
        print 'using kin'
    else:
        fname_ext=''
        backup_ext = '_kin'

    if task=='phaseBMI':
        targ_keys = [64, 68]

    else:
        targ_keys = key.lfp_targ

    n = []
    RT = dict()
    # for tg in targ_keys:
    #   RT[tg]=np.empty((1,))
    if 'anim' in kwargs.keys():
        anim = kwargs['anim']
    else:
        anim = 'seba'

    for d,day in enumerate(dates):
        for b,bl in enumerate(blocks[d][0]):
            fname = anim+day+bl+fname_ext+'.mat'
            fname_backup = anim+day+bl+backup_ext+'.mat'
            try:
                d = sio.loadmat(key.pre+fname,variable_names='Strobed')
                print 'loading kin'
            except:
                d = sio.loadmat(key.pre+fname_backup, variable_names='Strobed')

            print 'fname: ', fname
            strobed = d['Strobed']

            if task=='multi': #Get reward trials
                tsk_ind = np.nonzero(strobed[:,1]==rew_key)[0]

            elif task=='cue_flash': #Get init trials
                tsk_ind = np.nonzero(strobed[:,1]==key.mc_go)[0]

            elif scipy.logical_or(task == 'mc', task=='MC'): 
                tsk_ind = np.nonzero(strobed[:,1]==rew_key)[0]

            elif task=='phaseBMI':
                tsk_ind = np.nonzero(strobed[:,1]==rew_key)[0]

            init = np.empty((len(tsk_ind),))
            targ = np.empty((len(tsk_ind),))

            go_cue_time = np.empty((len(tsk_ind),))
            n.extend([len(tsk_ind)])

            for j,t_ind in enumerate(tsk_ind):
                if key.LFP_only:
                    start = np.max([t_ind - 4, 0])

                elif task=='multi':
                    start = np.max([t_ind - 8, 0])

                elif task=='cue_flash':
                    start = np.max([t_ind - 5, 0])

                elif task == 'mc':
                    start = np.max([t_ind - 4, 0])

                elif task == 'phaseBMI':
                    start = np.max([t_ind - 7, 0])

                tmp = strobed[start:t_ind,1] #Codes from trial
                
                try:
                    
                    init_ind = start+[i for i,t in enumerate(list(tmp)) if np.any(t==targ_keys)][-1]
                    init[j] = int(init_ind) #Target index in strobed
                    targ[j] = strobed[int(init[j]),1] #LFP target ID

                    if key.LFP_only:
                        go_cue_time[j] = strobed[int(t_ind)-2,0]
                    elif task=='phaseBMI':
                        go_cue_time[j] = strobed[int(t_ind)-3, 0]
                    else:
                        go_cue_time[j] = 0 #Generated in other functions

                except:
                    print goose
                    print 'help!'

            #Take diff between time entered into periphery (code 28) and init (codes 84-87): 
            rt = strobed[init.astype(int)+1,0] - strobed[init.astype(int),0]

            if 'RT_sequ' in locals():
                RT_sequ = np.hstack((RT_sequ,rt))
                targ_sequ = np.hstack((targ_sequ,targ))
                go_cue_sequ = np.hstack((go_cue_sequ, go_cue_time))
            else:
                RT_sequ = rt
                targ_sequ = targ
                go_cue_sequ = go_cue_time

            for t,tg in enumerate(targ_keys):
                targ_ind = np.nonzero(targ==tg)[0]
                if tg in RT.keys():
                    RT[tg] = np.hstack((RT[tg],rt[targ_ind]))
                else:
                    RT[tg] = rt[targ_ind]

        print 'done with ' + day +'!'
    
    if sequential and key.LFP_only:
        return RT_sequ, targ_sequ, n, go_cue_sequ
    
    elif sequential and task=='phaseBMI':
        return RT_sequ, targ_sequ, n, go_cue_sequ

    elif sequential:
        return RT_sequ, targ_sequ, n
    
    else:
        return RT, n


def plot_LFP_RT(RT,key,from_mat=False):
    targ_keys = key.lfp_targ
    fig = plt.figure()
    plt.hold(True)
    for i,tg in enumerate(targ_keys):
        if from_mat:
            plt.plot([i]*(len(RT[str(tg)][0,1:])),RT[str(tg)][0,1:],'b.')
            plt.plot(i,np.mean(RT[str(tg)][0,1:]),'rD',markersize=8)            
        else:
            plt.plot([i]*(len(RT[tg])-1),RT[tg][1:],'b.')
            plt.plot(i,np.mean(RT[tg][1:]),'rD',markersize=8)
    plt.xlim([-.5, 3.5])
    plt.xticks([0,1,2,3],['0','0.33','0.67','1.0'])
    plt.xlabel('LFP Target Postion: Fraction Beta Power of Total Power')
    plt.ylabel('Time to Reach (LFPcode - code28) (sec)')
    plt.title('LFP Cursor Mod with Reach After')

def get_MC_RTs(dates,blocks,key,sequential=False,task='multi',only_basics=False, 
    only_basics_plus_MC=False,**kwargs):
    '''
     Gets MC reaction time (go cue, to leave center)

     only_basics gets 'n','go_cue_sequ'
     '''
    if ('use_kin_strobed' in kwargs.keys() and kwargs['use_kin_strobed']):
        fname_ext='_kin'
    else:
        fname_ext=''


    if 'anim' in kwargs.keys():
        anim = kwargs['anim']
    else:
        anim = 'seba'

    if only_basics or only_basics_plus_MC:
        n=[]
        for d,day in enumerate(dates):
            for b,bl in enumerate(blocks[d][0]):
                fname = anim+day+bl+fname_ext+'.mat'
                d = sio.loadmat(key.pre+fname,variable_names='Strobed')
                strobed = d['Strobed']
                tsk_ind = np.nonzero(strobed[:,1]==9)[0]
                if task=='targ_jump':
                    init = tsk_ind - 5
                    
                elif task =='MC' or task=='mc':
                    init = tsk_ind - 4

                go_cue_time = strobed[init,0]

                if only_basics_plus_MC:
                    if task=='MC' or task=='mc' or task=='targ_jump':
                        mc_lab = strobed[init-2,1]
                    else: 
                        print 'no MC possible, add rest of tasks'
                #TODO, add rest of tasks

                n.extend([len(tsk_ind)])
                if 'go_cue_sequ' in locals():
                    go_cue_sequ = np.hstack((go_cue_sequ, go_cue_time))
                else:
                    go_cue_sequ = go_cue_time

                if only_basics_plus_MC:
                    if 'MC_Label' in locals():
                        MC_Label = np.hstack((MC_Label, mc_lab))
                    else:
                        try:
                            MC_Label = mc_lab
                        except:
                            print 'no mc_lab possible :('
        if only_basics_plus_MC:
            return go_cue_sequ, n, MC_Label
        else:
            return go_cue_sequ, n

    else:
        init_key = key.mc_go
        move_key = key.move_onset
        periph_key = key.enter_periph
        rew_key = key.rew
        #mc_targ_keys = key.mc_targ
        
        if task is 'targ_jump':
            mc_targ_keys = range(64,78)
        else:
            mc_targ_keys = range(64, 73)

        lfp_targ_keys = key.lfp_targ
        n = []
        RT1 = dict()
        RT2 = dict()
        # for mc in mc_targ_keys:
        #   for lfp in lfp_targ_keys:
        #       RT1[mc,lfp]=np.zeros((1,))
        #       RT2[mc,lfp]=np.zeros((1,))

        for d,day in enumerate(dates):
            for b,bl in enumerate(blocks[d][0]):

                fname = anim+day+bl+fname_ext+'.mat'
                d = sio.loadmat(key.pre+fname,variable_names='Strobed')
                strobed = d['Strobed']

                if task=='multi': #Get reward trials
                    tsk_ind = np.nonzero(strobed[:,1]==rew_key)[0]

                elif task=='cue_flash': #Get init trials
                    tsk_ind = np.nonzero(strobed[:,1]==key.mc_go)[0]

                elif task=='MC':
                    tsk_ind = np.nonzero(strobed[:,1]==rew_key)[0]

                elif task=='targ_jump':
                    tsk_ind = np.nonzero(strobed[:,1]==rew_key)[0]

                n.extend([len(tsk_ind)])
                init = np.empty((len(tsk_ind),))
                mc_targ = np.empty((len(tsk_ind),)) #List of MC targets by rew trial
                lfp_targ = np.empty((len(tsk_ind),)) #List of LFP targets by rew trial

                for j,t_ind in enumerate(tsk_ind):

                    #look for start of trial: 
                    tmp = strobed[0:t_ind+1,1]
                    try:
                        start = np.nonzero(tmp==key.trial_start)[0][-1]
                    except:
                        start = 0
                        print 'error in trial1, no starting code'

                    tmp = strobed[start:t_ind+1,1]
                    
                    if task is 'targ_jump':
                        if len(tmp) > 6:
                            jump_flag = 1
                            tmp1 = tmp[0:4]
                            tmp2 = tmp[2:]
                        else:
                            jump_flag = 0

    ##PARSE Target Jumps. 


                    try:
                        mc_targ_ind = start+[i for i,t in enumerate(list(tmp)) if np.any(t==mc_targ_keys)][0]
                        mc_targ[j] = strobed[int(mc_targ_ind),1]

                        if task is not 'MC':
                            lfp_targ_ind = start+[i for i,t in enumerate(list(tmp)) if np.any(t==lfp_targ_keys)][0]
                            lfp_targ[j] = strobed[int(lfp_targ_ind),1]

                        init_ind = start+[i for i,t in enumerate(list(tmp)) if t==init_key][0] #store init_key index (go cue)
                        init[j] = int(init_ind)

                    except:
                        print 'no mc label, no LFP label, ignore trials!'
                        init[j] = -1

                #rt1 is (leave_center_code '6' - go_cue_code '5')
                #rt2 is (enter_periph_code '7' - leave_center_code '6')

                #Remove all entries where init[j] = -1 but keep indices:
                good_trials = np.nonzero(init>-1)[0]
                bad_trials = np.nonzero(init==-1)[0]
                total_trials = len(init)

                init = init[init>-1]
                mc_targ[bad_trials] = 0
                lfp_targ[bad_trials] = 0
                tsk_ind[bad_trials] = 0

                if task == 'multi':
                    rt1 = strobed[init.astype(int)+1,0] - strobed[init.astype(int),0]
                    rt2 = strobed[init.astype(int)+2,0] - strobed[init.astype(int)+1,0]
                    trial_type_key = np.zeros(( len(tsk_ind), )) + 9

                elif task == 'cue_flash':
                    rt1 = strobed[init.astype(int)+2,0] - strobed[init.astype(int),0]
                    rt2 = strobed[init.astype(int)+3,0] - strobed[init.astype(int)+2,0]
                    trial_type_key = np.zeros(( len(tsk_ind), ))+9 #reward trials
                    err_trials = np.nonzero(strobed[tsk_ind+3, 1]==20)[0]
                    trial_type_key[err_trials] = 20 #error trials

                elif task == 'MC':
                    rt1 = strobed[init.astype(int)+1,0] - strobed[init.astype(int),0]
                    rt2 = strobed[init.astype(int)+2,0] - strobed[init.astype(int)+1,0]
                    trial_type_key = np.zeros(( len(tsk_ind), )) + 9

                rt1_full = rt2_full = trial_type_key_full = go_cue_time_full = np.zeros((total_trials, ))
                rt1_full[good_trials] = rt1
                rt2_full[good_trials] = rt2
                trial_type_key_full[good_trials] = trial_type_key
                go_cue_time_full[good_trials] = strobed[init.astype(int),0]
                
                if 'RT1_sequ' in locals():

                    RT1_sequ = np.hstack((RT1_sequ,rt1_full))
                    RT2_sequ = np.hstack((RT2_sequ,rt2_full))
                    mc_targ_sequ = np.hstack((mc_targ_sequ,mc_targ))
                    go_cue_sequ = np.hstack((go_cue_sequ, go_cue_time_full))
                    trial_type = np.hstack((trial_type, trial_type_key_full))
                    if task is not 'MC':
                        lfp_targ_sequ = np.hstack((lfp_targ_sequ,lfp_targ))
                else:
                    RT1_sequ = rt1_full
                    RT2_sequ = rt2_full
                    mc_targ_sequ = mc_targ
                    go_cue_sequ = go_cue_time_full
                    trial_type = trial_type_key_full
                    if task is not 'MC':
                        lfp_targ_sequ = lfp_targ

                if task is not 'MC':
                    for m,mc in enumerate(mc_targ_keys):
                        #All MC targets == tg
                        mc_targ_ind = np.nonzero(mc_targ==mc)[0]

                        for l,lfp in enumerate(lfp_targ_keys):
                            #All LFP targets == lfp
                            lfp_targ_ind = np.nonzero(lfp_targ==lfp)[0]

                            ind = set(mc_targ_ind) & set(lfp_targ_ind)
                            ind = np.array(list(ind))

                            if len(ind>0):
                                if (mc,lfp) in RT1.keys():
                                    RT1[mc,lfp] = np.hstack((RT1[mc,lfp],rt1_full[ind]))
                                    RT2[mc,lfp] = np.hstack((RT2[mc,lfp],rt2_full[ind]))
                                else:
                                    RT1[mc,lfp] = rt1_full[ind]
                                    RT2[mc,lfp] = rt2_full[ind]
                else:
                    lfp_targ_sequ = 0

            print 'done with ' + day +'!'
        if sequential:
            return RT1_sequ, RT2_sequ, mc_targ_sequ, lfp_targ_sequ, go_cue_sequ, n, trial_type
        else:
            return RT1, RT2, n

def plot_MC_RT_by_LFP(RT,key,type='separate_by_MC',title_str='Reaction Time'):
    plt.figure()
    mc_targ_keys = key.mc_targ
    lfp_targ_keys = key.lfp_targ
    
    if type == 'separate_by_MC':
        for m,mc in enumerate(mc_targ_keys):
            plt.subplot(2,2,m+1)
            plt.hold(True)
            cnt = 0
            for l,lfp in enumerate(lfp_targ_keys):
                plt.plot([l]*len(RT[mc,lfp][1:]), RT[mc,lfp][1:],'b.')
                plt.plot(l, np.mean(RT[mc,lfp][1:]),'rD',markersize=8)
                cnt += len(RT[mc,lfp][1:])
                plt.xticks([0,1,2,3],['0','0.33','0.67','1.0']) 
                plt.title('MC target: '+str(mc)+', n = '+str(cnt)+' '+title_str)
                plt.xlim([-.5, 3.5])
                plt.ylim([0, 0.7])
                plt.xlabel('LFP Targets')
                plt.ylabel('Time (sec)')
                plt.tight_layout()

    elif type == 'combine':
        for l,lfp in enumerate(lfp_targ_keys):
            plt.hold(True)
            cnt = 0
            for m,mc in enumerate(mc_targ_keys):
                if 'L' in locals():
                    L = np.hstack((L, RT[mc,lfp][1:]))
                else:
                    L = RT[mc,lfp][1:]

            plt.plot([l]*len(L), L, 'b.')
            plt.plot([l], np.mean(L),'rd')

        plt.xticks([0,1,2,3],['0','0.33','0.67','1.0']) 
        plt.title('MC RTs by LFP target, type = '+title_str)
        plt.xlim([-.5, 3.5])
        plt.ylim([0, 0.7])
        plt.xlabel('LFP Targets')
        plt.ylabel('Time (sec)')
        plt.tight_layout()

    elif type =='combine_hist':
        for m,mc in enumerate(mc_targ_keys):
            plt.subplot(2,2,m+1)
            plt.hold(True)
            cnt = 0
            tmp = []
            q = np.arange(0,1,20)
            for l,lfp in enumerate(lfp_targ_keys):
                tmp.extend(list(RT[mc,lfp][1:]))

            n,q = np.histogram(tmp,bins=q,normed=True)
            plt.bar(q[1:],n)
            #plt.xticks([0,1,2,3],['0','0.33','0.67','1.0'])    
            plt.title('MC target: '+str(mc)+', n = '+str(cnt)+' '+title_str)
                
                
            plt.xlabel(title_str)
            plt.ylabel('Frequency')
            plt.tight_layout()


def cursor_hist_by_LFP(dates,blocks,key,metric='cursor',lfp_only=False):
    '''
    function to tell what cursor actity looks 
    like while spec target is on the screen
    '''
    #Cursor activity goes from [-1, 1]
    bin = np.linspace(-1,1,2000)
    go_cue = key.mc_go
    targ_keys = np.array(key.lfp_targ)

    targ_hist_cnt = dict()

    for d,day in enumerate(dates):
        for b,block in enumerate(blocks[d][0]):
            strobed_ind = []
            strobed_num = []

            go_ind = []

            #Load cursor info:
            fname = '/Volumes/carmena/seba/seba'+day+'/bmi_data/dat'+day+block+'_bmi.mat'
            s = sio.loadmat(fname)
            if metric == 'cursor':
                cursor = s['data'][0,0]['lfp_cursor_kin'][0,:]
            elif metric == 'beta':
                cursor = np.mean(s['data'][0,0]['features'][-16:-12,:],axis=0)
            elif metric == 'total_pw':
                cursor = np.mean(s['data'][0,0]['features'][-4:,:],axis=0)

            events = s['data'][0,0]['events'][:,0]

            for t in np.arange(1,6001):
                if len(events[t])>0:
                    event_t = events[t][:,0]
                    for e,ev in enumerate(event_t):
                        strobed_ind.append(t)
                        strobed_num.append(ev)
                        if lfp_only:
                            if ev == key.lfp_only_go:
                                go_ind.append(len(strobed_num))
                        else:
                            if ev==go_cue:
                                #Num of strobed corresponding to go_cue
                                go_ind.append(len(strobed_num))
            #Find epochs of LFP cursor being on: (cursor num - GO cue): 

            target_num = []
            ind = np.zeros((len(go_ind),2))

            for g,go in enumerate(go_ind):
                if lfp_only:

                    if len(strobed_num)<=go:
                        print 'skipping last go_ind'
                    else:
                        target_num.append(strobed_num[go]) #Get target number for each 'Go Cue' ( 'go' is actually go_index + 1 )
                        ind[g,0] = strobed_ind[go-1] #Start of LFP target on = 'Go Cue (code = 15)'

                        #Reward 
                        tmp = strobed_num[go:] #codes after go cue
                        tmp_ind = np.nonzero(np.array(tmp)==key.lfp_only_rew)[0] #indices of all codes in strobed == reward after GO
                        
                        if len(tmp_ind)>0:
                            tmp_ind2 = tmp_ind[0] +go #take first one and add back 'go cue' index

                            #Check that no '4101s' in between
                            tmp3 = np.array(strobed_num[go:tmp_ind2])
                            
                            if np.any(tmp3==4101):
                                print 'pausing error! preeya, pay attention!'
                            else:
                                ind[g,1] = strobed_ind[tmp_ind2]

                else:
                    target_num.append(strobed_num[go-3])
                    ind[g,0] = strobed_ind[go-3]
                    ind[g,1] = strobed_ind[go-1]

            target_num = np.array(target_num)
            
            for t,targ in enumerate(set(target_num)):
                tind = np.nonzero(target_num==targ)[0]
                c = np.array([0])
                for i in range(len(tind)):
                    c = np.hstack((c, cursor [ ind[tind[i],0] : ind[tind[i],1] ]  ))

                if targ in targ_hist_cnt.keys():
                    targ_hist_cnt[targ] = np.hstack((targ_hist_cnt[targ], c))
                else:
                    targ_hist_cnt[targ] = c
    targ_hist_cnt['metric'] = metric
    return targ_hist_cnt

def plot_targ_hist_cnt(targ_hist_cnt,key):
        '''
        plots histogram for targets 
        '''
        col = ['r','b','g','m','c']
        fig, ax = plt.subplots()
        
        if targ_hist_cnt['metric'] == 'cursor':
            bin = np.linspace(-1,1,40)
        elif targ_hist_cnt['metric'] == 'beta':
            bin = np.linspace(0,7,40)
        elif targ_hist_cnt['metric'] == 'total_pw':
            bin = np.linspace(0,100,40)
        width = (bin[1]-bin[0])/5.
        X = np.zeros(( len(bin)-1,len(targ_hist_cnt.keys()[1:]) ))
        keys = set.intersection(set(targ_hist_cnt.keys()),set(key.lfp_targ))

        for i,t in enumerate(keys):
            x = targ_hist_cnt[t]
            n, bins = np.histogram(x,bins=bin)
            X[:,i] = n
            ax.plot(bins[1:],X[:,i]/float(np.sum(X[:,i])),'.-',color=col[i],label=str(t))
        plt.legend(loc=1)
        plt.xlabel('Cursor Position')
        plt.ylabel('Frequency')

def get_kinematic_features(dates,blocks,key,task='multi', shenoy_method = False, mc_lab = None, prep=False,
    return_go=False,**kwargs):

    NN = [77, 113, 109,  95,  79,  61,  69,  48,  75,  96,  68,  83, 105, 88]

    '''
     Gets MC reaction time (go cue, to onset of movement)
     Stores trials sequentially
    '''
    if ('use_kin_strobed' in kwargs.keys() and kwargs['use_kin_strobed']):
        fname_ext='_kin'
    else:
        fname_ext=''

    pre = key.pre
    init_key = key.mc_go
    move_key = key.move_onset

    rew_key = key.rew
    if 'anim' in kwargs.keys():
        anim = kwargs['anim']
    else:
        anim = 'seba'

    GO = []

    for d,day in enumerate(dates):
        for b,bl in enumerate(blocks[d][0]):

            fname = anim+day+bl+fname_ext+'.mat'
            print fname
            try:
                d = sio.loadmat(pre+fname,variable_names='Strobed')
            except:
                'cant load strobed?!'

            strobed = d['Strobed']
            periph_ind = np.nonzero(strobed[:,1]==7)[0]
            #inputs to get_kin_signal
            

            if task=='multi':
                tsk_ind = np.nonzero(strobed[:,1]==rew_key)[0]
                go_cue_sequ = strobed[tsk_ind-3,0]
                go_cue_ind = tsk_ind - 3

                #if len(go_cue_ind) != NN

                GO.extend(list(go_cue_sequ))
                mc_lab = np.zeros((len(go_cue_ind), ))

                for q, rew_trial_go in enumerate(go_cue_ind):
                    trial_init_set = np.nonzero(strobed[:rew_trial_go,1]==2)[0]
                    if len(trial_init_set)>0:
                        trial_init = trial_init_set[-1]
                        mc_lab[q] = strobed[trial_init+2, 1]
                    else:
                        print 'ignore trial'

            elif task =='cue_flash':
                tsk_ind = np.nonzero(strobed[:,1]==init_key)[0]
                go_cue_sequ = strobed[tsk_ind,0]
                go_cue_ind = tsk_ind
                mc_lab = ADD_HERE

            elif task in ['MC', 'mc']:
                tsk_ind = np.nonzero(strobed[:,1]==rew_key)[0]
                go_cue_sequ = strobed[tsk_ind-4,0]
                go_cue_ind = tsk_ind - 4
                mc_lab = strobed[tsk_ind - 6, 1]
                q = 0
            else:
                print 'wrong taskname!'
                print camel #error out
            
            #kin signal: 
            if mc_lab[q] >0 :
                if shenoy_method: 
                    if 'all_kins' in kwargs.keys():
                        kin_sig, kin_sig2, targ_dir_array = sm.get_sig([day], [[bl]], go_cue_sequ, [len(tsk_ind)],signal_type='shenoy_jt_vel',mc_lab = mc_lab, prep=prep,**kwargs)
                    else:
                        kin_sig, targ_dir_array = sm.get_sig([day], [[bl]], go_cue_sequ, [len(tsk_ind)],signal_type='shenoy_jt_vel',mc_lab = mc_lab, prep=prep,**kwargs)
                    
                    kin_feat = np.zeros((len(tsk_ind), 5 )) 


                    if kin_sig.shape[0] != len(go_cue_ind):
                        print "Incorrect return!", moose

            
                else:
                    kin_sig, targ_dir_array = sm.get_sig([day], [[bl]], go_cue_sequ, [len(tsk_ind)],signal_type='jt_vel',**kwargs)
                    kin_feat = np.empty((len(tsk_ind), 13 )) 
            
            #shape = rewards x time

            #V1: kinematic features :
            #0. max_speed
            #1. time of max_speed
            #2. 'slope' = onset (>0.17) to max
            #3. onset time (>0.17)
            #4. onset time (>0.7)
            #5 fast or slow: 1 = fast, 0 = slow
            
            #6. initial velocity: avg speed of onset to max
            #7. avg speed of onset to onset+400 ms
            #8. avg speed of max-400ms to max 
            #9. avg speed from onset to enter periph. 
            #10 sum speed: init - enter periph
            #11 range of speed: init - periph
            #12 periph time
            
            #V2: kinematic features: 
            #0. max_speed
            #1. time of max_speed
            #2. onset time (backwards from max, crosses 20%)
            #3. onset time (backwards from max, crosses 50% )
            #4. targ spec onset times (backwards from max, crosses 10% )
                    
            for r,tsk in enumerate(tsk_ind):

                if shenoy_method:
                    kin_feat =get_kin_sig_shenoy(kin_sig,anim=anim)
                    # start_time = 1200
                    # spd_after_go = kin_sig[r,start_time:]
                    # kin_feat[r,0] = np.max(spd_after_go)
                    # kin_feat[r,1] = start_time + np.argmax(spd_after_go)

                    # perc = [0.2, 0.5, 0.1]
                    # for p, per in enumerate(perc):
                    #   percent0 = kin_feat[r,0]*per #Bottom Threshold
                    #   indz = range(0,kin_feat[r,1].astype(int)+1-start_time) #0 - argmax_index
                    #   indzz = indz[-1:0:-1] #Reverse
                    #   datz = spd_after_go[indzz]
                    #   try:
                    #       x = np.nonzero(datz<percent0)[0][0]
                    #   except:
                    #       x = len(datz)
                    #   kin_feat[r,2+p] = kin_feat[r,1] - x
                    
                else:

                    per_ind = periph_ind[np.nonzero(periph_ind>go_cue_ind[r])[0][0]]
                    periph_tm = strobed[per_ind,0]
                    periph_tm = int((periph_tm - go_cue_sequ[r])*1000)

                    start_time = 1200
                    spd_after_go = kin_sig[r,start_time:]

                    kin_feat[r,0] = np.max(spd_after_go) #1. max_speed
                    kin_feat[r,1] = np.argmax(spd_after_go)+start_time #2. time of max_speed

                    tmp = np.nonzero(spd_after_go>0.17)[0]
                    if len(tmp):
                        onset1 = np.nonzero(spd_after_go>0.17)[0][0]+start_time 
                    else:
                        onset1 = start_time

                    onset2 = np.nonzero(spd_after_go>0.7)[0]
                    if len(onset2)>0:
                        onset2 = onset2[0]+start_time
                        fast = 0
                    else:
                        onset2 = np.NaN
                        fast = 1

                    kin_feat[r,3] = onset1 #3. onset 
                    kin_feat[r,4] = onset2 #4. onset 2
                    kin_feat[r,5] = fast #fast or slow

                    adj_onset = int(onset1-start_time)
                    adj_max = int(kin_feat[r,1] - start_time)
                    adj_periph = periph_tm + 300 #periph_tm calc wrt go cue (1500), so must add (1500-1200 = 300) back to correct 
                    print adj_onset, adj_periph

                    kin_feat[r,6] = np.mean(spd_after_go[adj_onset:adj_max]) #mean veloc. init - max
                    kin_feat[r,7] = np.mean(spd_after_go[adj_onset:np.min([adj_onset+400, 3000])]) #mean veloc. init - (init+400)
                    kin_feat[r,8] = np.mean(spd_after_go[np.max([adj_max-400, 0]):adj_max]) #mean veloc. (max-400) - max
                    
                    if adj_onset<adj_periph:
                        kin_feat[r,9] = np.mean(spd_after_go[adj_onset:adj_periph]) #mean veloc. init - enter periph
                        kin_feat[r,10] = np.sum(spd_after_go[adj_onset:adj_periph]) #integral? 
                        kin_feat[r,11] = np.max(spd_after_go[adj_onset:adj_periph]) - np.min(spd_after_go[adj_onset:adj_periph]) #range?
                    else:
                        kin_feat[r,9] = np.NaN
                        kin_feat[r,10] = np.NaN
                        kin_feat[r,11] = np.NaN

                    kin_feat[r,12] = periph_tm

                    #slope: 
                    max_ind = np.argmax(spd_after_go)
                    onset1 = onset1 - start_time

                    slope = ( spd_after_go[max_ind] - spd_after_go[onset1] ) / (.001*(max_ind - onset1))
                    kin_feat[r,2] = slope 

            if 'KIN_FEAT' in locals():
                KIN_FEAT = np.vstack((KIN_FEAT, kin_feat))
                KIN_SIG = np.vstack(( KIN_SIG, kin_sig))
                if 'all_kins' in kwargs.keys():
                    KIN_SIG2 = np.hstack((KIN_SIG2, kin_sig2))
    
            else:
                KIN_FEAT = kin_feat
                KIN_SIG = kin_sig
                if 'all_kins' in kwargs.keys():
                    KIN_SIG2 = kin_sig2

        print 'done with ' + day +'!'

    # d = dict()
    # d['max_speed'] = KIN_FEAT[:,0]
    # d['time_max_speec'] = KIN_FEAT[:,1]
    # d['slope'] = KIN_FEAT[:,2]
    # d['onset1'] = KIN_FEAT[:,3]
    # d['onset2'] = KIN_FEAT[:,4]
    # d['fast_trial'] = KIN_FEAT[:,5]
    if return_go:
        return KIN_FEAT, KIN_SIG, shenoy_method, GO
    else:
        if 'all_kins' in kwargs.keys():
            return KIN_FEAT, KIN_SIG, KIN_SIG2, shenoy_method
        else:
            return KIN_FEAT, KIN_SIG, shenoy_method


def bmi3d_get_1_4_targ_feats(task_name,pref='/Volumes/TimeMachineBackups/Cart/'):
    c = sio.loadmat(pref+task_name+'cart_cursor_all_task_dict.mat')
    t = sio.loadmat(pref+task_name+'cart_target_all_task_dict.mat')
    spec = sio.loadmat(pref+task_name+'cart_chan128_all_task_dict.mat')
    
    curs_b = np.arange(0, 4, 1/60.0)

    go = squeeze(t['go_cue_ind'])

    N_cumsum = np.hstack((0, np.cumsum(squeeze(t['n'])) ))

    n = squeeze(t['n'])
    N_nonrep = np.zeros((10, 10)) - 1
    REW_RATE = np.zeros((10, 1000)) - 1
    HOLD_ERR_RATE = np.zeros((10, 1000)) - 1
    TARG_CNT = np.zeros((10, 1000)) - 1
    
    window = 30*60

    for i, n_start in enumerate(N_cumsum[:-1]):
        go_blk = go[n_start:N_cumsum[i+1]]
        for j, r in enumerate(go_blk):
            x = np.nonzero(scipy.logical_and(go_blk>(r-window), go_blk<(r+window)))[0]
            REW_RATE[i,j] = len(x)

        TARG_CNT[i,:n[i]] = np.arange(n[i])
        N_nonrep[i,0] = n[i]

    #Now get mc_label, n , abs_time_o

    #Get cursor velocity: 
    fs = 60. #HZ. 

    vel = np.diff(c[task_name],axis=1)/(1000/fs)
    filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=1)

    vel_bins = curs_b[:-1] + 0.5*(curs_b[1] - curs_b[0])

    mc_vect = normalize(t[task_name], axis=1)
    mc_vect_mat = np.tile(mc_vect[:,np.newaxis,:], (1, filt_vel.shape[1], 1))

    #Now get kin_sig, kin_feat
    KIN_SIG = np.sum(np.multiply(mc_vect_mat, filt_vel), axis=2)
    start_bin = np.argmin(np.abs(curs_b-2))
    t_range= [2.5, 1.5]
    kin_feat = get_kin_sig_shenoy(KIN_SIG, bins=vel_bins, start_bin=start_bin,first_local_max_method=True)

    #SPEC: 


    ###Return mc_label, n, abs_time_go
    epsilon = 10**-2
    mc_label = np.zeros(( mc_vect.shape[0], ))
    mc_label_dict = dict()
    mc_label_dict['vector'] = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    mc_label_dict['label'] = np.array([64, 66 ,68, 70])
    for i, mv in enumerate(mc_vect):    
        tmp = np.sum( np.square( mc_label_dict['vector'] - np.tile(mv[np.newaxis,:], (4, 1)) ), axis= 1)
        idx = np.argmin(tmp)
        if tmp[idx] < epsilon:
            mc_label[i] = mc_label_dict['label'][idx]

    #Store all, similar to below: 
    sequ_RT = dict()
    sequ_RT['trial_type'] = np.array([0])
    sequ_RT['MC_label'] = mc_label
    sequ_RT['n'] = n
    sequ_RT['abs_time_go'] = go
    sio.savemat(pref+task_name+'-bmi3d_sequRT.mat',sequ_RT)

    kin_mets = dict()
    kin_mets['N_nonrep'] = N_nonrep
    kin_mets['kin_sig'] = KIN_SIG
    kin_mets['cursor_bin'] = curs_b
    kin_mets['kin_feat_bins'] = vel_bins
    kin_mets['kin_feat'] = kin_feat
    #kin_mets['non_adj_kin_feat'] = kin_feat
    kin_mets['targ_cnt'] = TARG_CNT 
    kin_mets['rew_rate'] = REW_RATE 
    sio.savemat(pref+task_name+'-bmi3d_kin_mets.mat',kin_mets)

    lat = sio.loadmat(pref+task_name+'cart_latchn_all_task_dict.mat')
    lat_spec = dict()
    lat_spec['spec'] = lat['spec'][np.newaxis, :, :, :]
    lat_spec['freq'] = squeeze(lat['freq'])
    lat_spec['bins'] = squeeze(lat['bins'])
    lat_spec['moving_window'] = np.array([0.201, 0.011])
    lat_spec['task'] = task_name
    sio.savemat(pref+task_name+'-bmi3d_sequRT_spec_win0_LAT.mat',lat_spec)

    spec_dict = dict()
    spec_dict['spec'] = spec['spec'][np.newaxis, :, : , :]
    spec_dict['freq'] = spec['freq'][0,:]
    spec_dict['bins'] = spec['bins'][0,:]
    spec_dict['moving_window'] = np.array([0.201, 0.011])
    spec_dict['task'] = task_name
    sio.savemat(pref+task_name+'-bmi3d_sequRT_spec_win0.mat',spec_dict)

def get_1_4_targ_feats(dates, blocks, key, task='MC',one_or_four=1,ignore=None):
    '''
     - Gets target count number for blocks of 1-target series
     - Also gets hit rate
    '''
    pre = key.pre
    init_key = key.mc_go
    move_key = key.move_onset
    rew_key = key.rew

    N = np.zeros((10, 10)) - 1
    REW_RATE = np.zeros((10, 1000)) - 1
    TARG_CNT = np.zeros((10, 1000)) - 1
    blk_cnt = 0

    for d,day in enumerate(dates):
        for b,bl in enumerate(blocks[d][0]):

            fname = 'seba'+day+bl+'.mat'
            try:
                str_file = sio.loadmat(pre+fname,variable_names='Strobed')
            except:
                print 'cant load strobed?!'
            strobed = str_file['Strobed']
            rew_ind = np.nonzero(strobed[:,1]==rew_key)[0]
            mc_label = strobed[rew_ind - 6, 1]

            #Remove repeats: 
            non_rep_trial = np.nonzero(mc_label < 100)[0]
            mc_label = mc_label[non_rep_trial]
            rew_ind = rew_ind[non_rep_trial]
            
            #Target Index in Block: 
            if one_or_four == 1:

                #This is for blocks with more than 1 target in each block (when I did 4 targs in a row)
                mc_set = list(set(mc_label))

                #For sets greater than 1
                if len(mc_set) > 1:
                    mc = mc_label[0]
                    cnt = 0
                    n = []

                    targ_cnt = np.zeros(( mc_label.shape[0], ))

                    for j, reach in enumerate(mc_label):

                        if j+1 == mc_label.shape[0]:
                            #Last index!
                            n.extend([cnt+1])
                            print 'J = ', j

                        if reach == mc:
                            targ_cnt[j] = cnt

                        elif reach != mc: 
                            #New target or last target
                            #Add old culmination to n: 
                            print 'J = ', j
                            n.extend([cnt])
                            mc = reach
                            cnt = 0
                            targ_cnt[j] = cnt

                        cnt += 1
                else:
                    targ_cnt = np.arange(mc_label.shape[0])
                    n = [mc_label.shape[0]]
            else: #If a 4 target set: 
                n = [mc_label.shape[0]]
                targ_cnt = np.arange(mc_label.shape[0])


            #Get Reward Rate: 
            rew_rate = np.zeros(( mc_label.shape[0], ))
            rew_times = strobed[rew_ind,0]

            for j, ridx in enumerate(rew_ind): #use rew_ind from before
                start = np.max([rew_times[j]-30, 0])
                stop = np.min([rew_times[j]+30, strobed[-1,0]])
                slc = np.nonzero(scipy.logical_and(rew_times>start, rew_times<stop))[0]
                rew_rate[j] = slc.shape[0]

            N[blk_cnt,:len(n)] = n
            REW_RATE[blk_cnt, :sum(n)] = rew_rate
            TARG_CNT[blk_cnt, :sum(n)] = targ_cnt

            
        
            #Remove ignored segments: 
            if ignore is not None: 
                if day in ignore[0]:
                    d_i = np.nonzero(np.array(ignore[0])==day)[0][0]
                    if bl in ignore[1][d_i][0]:
                        b_i = ignore[1][d_i][0].find(bl)

                        bl_ignore = ignore[2]

                        n_cumsum = np.hstack(( np.array([0]), np.cumsum(n) ))
                        #Reset N count: 
                        for bli, blinore in enumerate(bl_ignore):
                            n_bli = np.arange(n_cumsum[blinore-1], n_cumsum[blinore])

                            N[blk_cnt,blinore] = -1
                            REW_RATE[blk_cnt, n_bli] = 0
                            TARG_CNT[blk_cnt, n_bli] = 0


            blk_cnt += 1

    return N, REW_RATE, TARG_CNT
                

def get_kin_sig_shenoy(kin_sig, bins=np.linspace(0,3000,3000), start_bin=1200,first_local_max_method=False,
    after_start_est=300+300, kin_est = 1000, anim='seba'):

    kin_feat = np.zeros((kin_sig.shape[0], 5))

    for r in range(kin_sig.shape[0]):   
        spd_after_go = kin_sig[r,start_bin:]

        if first_local_max_method: #Done only on BMI 3d, assuming Fs = 60 Hz. 
            d_spd = np.diff(spd_after_go)

            #Est. number of bins RT should come after: 
            aft = after_start_est/float(1000)*60 #Aft is in iteration for bmi3d
            rch = kin_est/float(1000)*60 #rch is in iteration for bmi3d
            #Find first cross from + --> -
            
            max_ind = np.array([i for i, s in enumerate(d_spd[:-1]) if scipy.logical_and(s>0, d_spd[i+1]<0)]) #derivative crosses zero w/ negative slope
            z = np.nonzero(scipy.logical_and(max_ind>aft, max_ind<(rch+aft)))[0] #local maxima that fit estimate of rxn time --> rch time

            #How to choose: 
            if len(z)>0:
                z_ind = np.argmax(spd_after_go[max_ind[z]]) #choose the biggest
                kin_feat[r,1] = bins[max_ind[z[z_ind]]+start_bin] #write down the time
                maxbin = max_ind[z[z_ind]]+start_bin
            else:
                print ' no local maxima found within range :/ '
                kin_feat[r,1] = bins[int(start_bin+aft+rch)]
                maxbin = start_bin+aft+rch
        else:
            kin_feat[r,1] = bins[ start_bin + np.argmax(spd_after_go) ]
            maxbin = start_bin + np.argmax(spd_after_go)

        kin_feat[r,0] = kin_sig[r,int(maxbin)]

        perc = [0.2, 0.5, 0.1]
        for p, per in enumerate(perc):
            percent0 = kin_feat[r,0]*per #Bottom Threshold
            indz = range(0, int(maxbin-start_bin)) #0 - argmax_index
            indzz = indz[-1:0:-1] #Reverse
            datz = spd_after_go[indzz]
            try:
                x = np.nonzero(datz<percent0)[0][0]
            except:
                x = len(datz)
            kin_feat[r,2+p] = bins[int(maxbin-x)]
    return kin_feat

def _test_kin_features(pref='/Volumes/TimeMachineBackups/Cart2D/'): 
    S1 = sio.loadmat(pref+'S1-bmi3d_kin_mets.mat')
    K = S1['kin_feat']
    Sig = S1['kin_sig']
    bins = S1['kin_feat_bins'][0,:]

    ix = [i for i in range(10,618) if K[i,2]==K[i,4]]
    cmap = ['blue','pink', 'purple', 'red']
    for i in ix:
        plt.plot(bins, Sig[i,:],'k-')
        for j in [1,2,3,4]:
            t = K[i,j]
            bix = np.argmin(np.abs(bins-t))
            plt.plot(K[i, j], Sig[i,bix], '.', color=cmap[j-1],markersize = 20)
        plt.show()
        next = int(input('Next one? '))
        if next !=1:
            break

def squeeze(array):
    x,y = array.shape
    if x==1:
        new_array = array[0,:]
    elif y==1:
        new_array = array[:,0]
    else:
        new_array = array
    return new_array





