from tables import *
import tables
from sklearn.preprocessing import normalize
import numpy as np
import sav_gol_filt as sg_filt
import psycho_metrics as pm
import scipy.io as sio
import dbfunctions as dbfn
import fcns
import extract_cart_sdh as ecsdh
from chance_1D_bmi3dLFP import calc_chance
import datetime
import multiprocessing

#Reaching behavior class and neural behavior class
from make_pytable_reach_bmi3d import manual_control_data
from make_pytable_reach_bmi3d import Behav_Hand_Reach, Neural_Hand_Reach, Kin_Traces

def get_task_entries(fname='task_entries_lfpmod_jan15.mat',
	tasks=['lfp_mod', 'lfp_mod_plus_mc', 'lfp_mod_mc_reach_out'],
	system='sdh', **kwargs):

	task_entry_dict=dict()
	for tsk in tasks:
		if 'startdate' in kwargs.keys():
			startdate = kwargs['startdate']
		else:
			startdate = datetime.date(2015,01,01)
		if 'enddate' in kwargs.keys():
			enddate = kwargs['enddate']
		else:
			enddate = datetime.date.today()

		if 'te_max' in kwargs.keys():
			te_max = kwargs['te_max']
		else:
			te_max = None

		if 'te_min' in kwargs.keys():
			te_min = kwargs['te_min']
		else:
			te_min = None

		te = ecsdh.get_ids(tsk,startdate=startdate, enddate=enddate, min_task_len=5.,
			te_max = te_max,te_min=te_min,system=system)
		
		task_entry_dict[tsk] = te
	sio.savemat(fname, task_entry_dict)

class Behav_LFPMod_TE(IsDescription):
	trial_type = StringCol(256) #'lfp_mod', 'lfp_hld', 'lfp_rch'
	task_entry = IntCol()#task entry number (xxxx)
	frac_lims = Float64Col(shape=(2,)) #same for whole block
	powercap = Float64Col() #same for whole block -- only relevant after 1/8
	nsteps = Float64Col() #in decoder
	chance_arr = Float64Col(shape=(100,)) #same for whole block
	tot_rew = Float64Col()

class Behav_LFPMod_Trial(IsDescription):
	task_entry = IntCol()
	start_time = IntCol()
	lfp_targ_loc = Float64Col() #z axis coord of target
	reach_time = Float64Col() #wait --> enter LFP_target
	hold_time = Float64Col() #enter LFP_target --> reward (or reach)
	reach_err = Float64Col() #sum of (lfp_pos-targ_pos) from wait--> reward (or reach)
	power_err_cnt = IntCol() #number of power errors in trial
	hold_err_cnt = IntCol() #number of hold errors in trial
	rew_rate = Float64Col() #rew / min

class Behav_Hand_Reach_LFP(IsDescription):
	target_loc_mc = Float64Col(shape=(2,))
	target_loc_lfp = Float64Col(shape=(2,))
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

class Kin_Traces_LFP(IsDescription):
	cursor_traj = Float64Col(shape=(int(2000*(60./1000)), 2))
	mc_vect = Float64Col(shape=2,)
	kin_sig = Float64Col(shape=(int(2000*(60./1000))-1, ))
	kin_feat = Float64Col(shape=5, )
	task_entry = IntCol()
	start_time = IntCol()

class lfp_task_data(manual_control_data):
	def __init__(self, *args, **kwargs):
		super(lfp_task_data, self).__init__(*args,**kwargs)
		self.tasks = ['lfp_mod','lfp_mod_plus_mc','lfp_mod_mc_reach_out']
		self.behav_fname_kin= 'kin'+self.behav_fname

	def get_behavior(self):
		##only for lfp_mc_plus_reach_out task: 
		h5file = openFile(self.tdy_str + self.behav_fname_kin +'lfp_mod_mc_reach_out'+'.h5', mode="w", title='Cart, lfp_mc_behavior')
		table_trial = h5file.createTable("/", 'trl_bhv', Behav_Hand_Reach_LFP, "Trial Behavior Table")
		kin_table = h5file.createTable("/",'trl_kin',Kin_Traces_LFP,"Trial Kin Table")

		tsk_te = self.task_entry_dict['lfp_mod_mc_reach_out']
		t1, t2 = tsk_te.shape
		if t1==1 and t2==1:
			tsk_te = np.array([tsk_te[0,0]])
		else:
			tsk_te = np.squeeze(tsk_te)

		for j, te in enumerate(tsk_te):
			task_entry = dbfn.TaskEntry(te)
			nm = task_entry.name
			hdf = tables.openFile('/storage/bmi3d/rawdata/hdf/'+nm+'.hdf')
			
			rew_ind = np.array([i for i, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
			mc_targ_ind = np.array([i for i, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='mc_target'])

			msg = hdf.root.task_msgs[:]['msg']
			msg_time = hdf.root.task_msgs[:]['time']
			rew_times = msg_time[rew_ind]

			for i, trial in enumerate(rew_ind):
				trial_row = table_trial.row
				kinrow = kin_table.row

				trial_row['trial_outcome'] = 'reward'

				targ_ind = msg_time[trial-2]
				trial_row['target_loc_mc'] = hdf.root.task[targ_ind]['mc_targ'][[0,2]]

				targ_ind2 = msg_time[trial-4]
				trial_row['target_loc_lfp'] = hdf.root.task[targ_ind2]['lfp_target'][[0,2]]

				reach_time = (msg_time[trial-1] - msg_time[trial-2])*(1000./60.)
				trial_row['reach_time'] = reach_time

				go_time = msg_time[trial-2]
				trial_row['go_time'] = go_time

				trial_row['start_time'] = msg_time[mc_targ_ind[i]]					
				trial_row['task_entry'] = te #for synching later

				kinrow['start_time'] = msg_time[mc_targ_ind[i]]	
				kinrow['task_entry'] = te#for synching later

				#Kinematics; 
				start_time = go_time - 200.*(60./1000)
				cursor = hdf.root.task[start_time-(500.*(60./1000)):start_time+(1500.*(60./1000))]['cursor'][:,[0,2]]
				curs_b = np.arange(-500,1500,1000./60.)
				vel = np.diff(cursor,axis=0)/(1000/60.)
				filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=0)
				vel_bins = curs_b[:-1] + 0.5*(curs_b[1] - curs_b[0])

				targ = hdf.root.task[targ_ind+2]['mc_targ'][[0,2]]
				assert not np.all(targ==0.) #Target is not at the origin
				mc_vect = targ/np.sum(targ)
				mc_vect_mat = np.tile(mc_vect[np.newaxis,:], (filt_vel.shape[0], 1))

				#Now get kin_sig, kin_feat
				KIN_SIG = np.sum(np.multiply(mc_vect_mat, filt_vel), axis=1)
				start_bin = int(np.argmin(np.abs(curs_b)) - 300*(60/1000.)) #start bin is 300 ms before go cue
			
				kin_feat = pm.get_kin_sig_shenoy(KIN_SIG[np.newaxis], bins=vel_bins, start_bin=start_bin,
					first_local_max_method=True) #returns signal in terms of bins

				rxn_time = kin_feat[0,2]

				kinrow['cursor_traj'] = cursor
				kinrow['mc_vect'] = mc_vect
				kinrow['kin_sig'] = KIN_SIG
				kinrow['kin_feat'] = kin_feat[0,:]

				trial_row['rxn_time'] = rxn_time


				trial_row.append()
				kinrow.append()
			kin_table.flush()
			table_trial.flush()
		h5file.close()

	def get_lfpmod_behavior(self,system='sdh'):
		TE = self.task_entry_dict
		TE_rew_inds = dict()
			
		# for j, te in enumerate(np.squeeze(TE['task_entries'])):

		# 	#### BLOCK METRICS ####

		for k, tsk in enumerate(self.tasks):
			h5file = openFile(self.tdy_str + self.behav_fname +tsk+'.h5', mode="w", title='Cart, lfp_behavior')
			table_block = h5file.createTable("/", 'blk_bhv', Behav_LFPMod_TE, "Block Behavior Table")
			table_trial = h5file.createTable("/", 'trl_bhv', Behav_LFPMod_Trial, "Trial Behavior Table")

			for j, te in enumerate(np.squeeze(TE[tsk])):
				block = table_block.row
				task_entry = dbfn.TaskEntry(te)
				block['trial_type'] = task_entry.task.name
				block['frac_lims'] = task_entry.params['lfp_frac_lims']
				if 'powercap' in task_entry.params.keys():
					block['powercap'] = task_entry.params['powercap']
				block['task_entry'] = te #

				#load n_steps: 
				#import pickle
				dec_steps =  task_entry.decoder_filename[-11:-10]
				try:
					block['nsteps'] = int(dec_steps)
				except:
					block['nsteps'] = -1
				# try:
				# 	dec = pickle.load(open('/storage/bmi3d/decoders/'+dec_fname))
				# except:
				# 	dec = pickle.load(open('/home/preeya/decoders/'+dec_fname))

				#block['nsteps'] = task_entry.decoder.filt.n_steps
				#Get reward trials only for LFP 

				nm = task_entry.name
				if system == 'sdh':
					hdf = tables.openFile('/storage/bmi3d/rawdata/hdf/'+nm+'.hdf')
				elif system == 'arc' or 'nucleus':
					hdf = tables.openFile('/storage/rawdata/hdf/'+nm+'.hdf')
				
				#Trial Starts: 
				rew_ind = np.array([i for i, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
				block['tot_rew'] = len(rew_ind)

				#Get chance
				rew , ri, tl, act_rew = calc_chance(te,system=system)
				block['chance_arr'] = rew

				block.append()
				
				#### TRIAL METRICS ####
				msg = hdf.root.task_msgs[:]['msg']
				msg_time = hdf.root.task_msgs[:]['time']


				if task_entry.task.name in [u'lfp_mod', u'lfp_mod_plus_mc']:
					rew_ind = np.array([j for j, i in enumerate(msg) if i=='reward'])

				elif task_entry.task.name in [u'lfp_mod_mc_reach_out']:
					rew_ind = np.array([j for j, i in enumerate(msg) if i=='mc_target'])
				else:
					print 'unrecognized task!'

				rew_times = msg_time[rew_ind]
				TE_rew_inds[str(task_entry.task.name), te] = rew_times

				#Categorize trials: 
				for trial_ind, ix in enumerate(rew_ind):

					#Get Row of table: 
					trial = table_trial.row
					trial['task_entry'] = te #for synching later
					trial['start_time'] = rew_times[trial_ind]

					#Find start of trial: 
					ind_wait = ix-1
					while (msg[ind_wait] not in ['wait']):
						ind_wait -=1
					
					#Start_time = first 'lfp_target'
					ind_lfp_targ = ind_wait.copy()
					while (msg[ind_lfp_targ] not in ['lfp_target']):
						ind_lfp_targ += 1

					start_time = msg_time[ind_lfp_targ]
					
					#Enter target time (lfp_hold) before reward
					enter_time = msg_time[ix-1] 

					trial['reach_time'] = (enter_time - start_time)*(1000./60.)
					trial['hold_time'] = (msg_time[ix] - enter_time)*(1000./60.)

					trial_msg = msg[ind_lfp_targ:ix]
					trial['power_err_cnt'] = len([i for i,j in enumerate(trial_msg) if j=='powercap_penalty'])
					trial['hold_err_cnt'] = len([i for i,j in enumerate(trial_msg) if j=='hold_error'])

					#Get lfp_cursor things: #REMEMBER, now switching to LFP task timing , so need rew_TIMEs, not rew_IND

					lfp_targ_location = hdf.root.task[start_time]['lfp_target'][2]
					trial['lfp_targ_loc'] = lfp_targ_location
					lfp_traj = hdf.root.task[start_time:enter_time]['lfp_cursor'][:,2]

					trial['reach_err'] = np.sum(np.abs(lfp_traj - lfp_targ_location))

					#Reward Rate: 
					trial_time = msg_time[ix]
					end_session_time = msg_time[-1]

					if trial_time < 0.5*(60*60):
						rews = np.nonzero(rew_times<trial_time+(0.5*(60*60)))[0]
						rew_rate = len(rews)/float(0.5+ trial_time/(60*60.))
					elif trial_time + 0.5*(60*60) > end_session_time:
						rews = np.nonzero(rew_times>trial_time-(0.5*(60*60.)))[0]
						rew_rate = len(rews)/float(0.5+ (end_session_time - trial_time)/(60*60))
					else:
						rix, rval = fcns.get_in_range(rew_times, [trial_time - (0.5*(60.*60)), trial_time + (0.5*(60.*60))])
						rew_rate = len(rix)
					
					trial['rew_rate'] = rew_rate
					trial.append()
				block.append()
				table_block.flush()
				table_trial.flush()
			h5file.close()
		sio.savemat(self.task_entry_dict_time_inds_fname, TE_rew_inds)
		self.task_entry_dict_time_inds = TE_rew_inds

if __name__ == "__main__":

	kw = dict(te_min=6927, startdate=datetime.date(2015,03,28))
	get_task_entries(fname='task_entries_lfpmod_apr15_rev_targ1.mat',**kw)

	d = dict(behav_file_name='t1_low_beta_lfp_cart_behav',\
		neural_file_name = 't1_low_beta_lfp_cart_neural',\
		task_entry_dict_fname='task_entries_lfpmod_apr15_rev_targ1.mat',\
		task_entry_dict_go = 'task_entries_lfpmod_apr15_rev_targ1_go.mat',\
		t_range=[2.5, 1],\
		)


	# kw = dict(te_min=6484, te_max=6532)
	# get_task_entries(fname='task_entries_lfpmod_jan15_low_beta_targ1.mat',**kw)
	# d = dict(behav_file_name='t1_low_beta_lfp_cart_behav',\
	# 	neural_file_name = 't1_low_beta_lfp_cart_neural',\
	# 	task_entry_dict_fname='task_entries_lfpmod_jan15_low_beta_targ1.mat',\
	# 	task_entry_dict_go = 'task_entries_lfpmod_jan15_low_beta_targ1.mat',\
	# 	t_range=[2.5, 1],\
	# 	)


	# kw = dict(startdate=datetime.date(2015,01,28),enddate=datetime.date(2015,01,30),te_min = , te_max = 6483)
	# get_task_entries(fname='task_entries_lfpmod_jan15_low_beta_targ3.mat',**kw)

	# d = dict(behav_file_name='t3_low_beta_lfp_cart_behav',\
	# 	neural_file_name = 't3_low_beta_lfp_cart_neural',\
	# 	task_entry_dict_fname='task_entries_lfpmod_jan15_low_beta_targ3.mat',\
	# 	task_entry_dict_go = 'task_entries_lfpmod_jan15_low_beta_targ3.mat',\
	# 	t_range=[2.5, 1],\
	# 	)

	lfpd = mp.lfp_task_data(**d)

	#Get behav: 
	lfpd.get_lfpmod_behavior()

	lfpd.get_behavior()

	#Get neural:
	jobs = []
	kw = dict(t_range=[2.5, 1])
	lfpd.tasks = ['lfp_mod_mc_reach_out'] 
	for tsk in lfpd.tasks:
		print 'tsk: ', tsk
		p = multiprocessing.Process(target=lfpd.make_neural,args=([tsk],),kwargs=kw)
		jobs.append(p)
		p.start()
		#mcd.make_neural(tsk)


