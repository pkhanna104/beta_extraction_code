import numpy as np
import sav_gol_filt as sg_filt
import psycho_metrics as pm
import fcns

def get_CO_metrics(start_ind, msg, msg_time, trial, kinrow, TE_go_inds, rew_times, hdf,tsk, te):
	trial_ind = start_ind[0]
	ix = start_ind[1]

	# Get trial outcome
	if msg[ix+2] == 'timeout_penalty':
		trial_outcome = 'center_timeout'
	elif msg[ix+3] == 'hold_penalty':
		trial_outcome = 'center_holderr'
	elif msg[ix+5] == 'timeout_penalty':
		trial_outcome = 'periph_timeout'
	elif msg[ix+6] == 'hold_penalty':
		trial_outcome = 'periph_holderr'
	elif msg[ix+7] == 'reward':
		trial_outcome = 'reward_trial__'
	else:
		print 'error: unrecognized trial type, msg index: ', ix
	trial['trial_outcome'] = trial_outcome

	if trial_outcome not in ['center_timeout', 'center_holderr']:
		#Get go time: 
		go_time = msg_time[ix+4]
		TE_go_inds[tsk,te].append(go_time)
	else:
		go_time = -1
		TE_go_inds[tsk,te].append(0)

	trial['go_time'] = go_time

	#Get Reach time: 
	if trial_outcome in ['periph_holderr', 'reward_trial__']:
		assert (msg[ix+4] == 'target' and msg[ix+5] =='hold')
		reach_time = (msg_time[ix+5] - msg_time[ix+4])*(1000./60.)
	else:
		reach_time = -1

	trial['reach_time'] = reach_time

	#Get Rxn time and reach error: 
	if trial_outcome in ['periph_holderr', 'reward_trial__']:
		#Cursor trajectory: 
		start_time = go_time - 200.*(60./1000)

		if start_time < int(500.*(60./1000)):
			foo = np.arange(-500., 1500., 1000/60.)
			cursor = np.zeros((len(foo), 2))
			cursor_trunc = hdf.root.task[:start_time+(1500.*(60./1000))]['cursor'][:,[0,2]]
			cursor[len(cursor)-cursor_trunc.shape[0]:,:] = cursor_trunc
		
		elif start_time + (1500.*(60./1000)) > hdf.root.task[:]['cursor'].shape[0]:
			foo = np.arange(-500., 1500., 1000/60.)
			cursor = np.zeros((len(foo), 2))
			cursor_trunc = hdf.root.task[start_time-(500.*(60./1000)):]['cursor'][:,[0,2]]
			cursor[:cursor_trunc.shape[0],:] = cursor_trunc
		
		else:#Really (-.7 : 1.3)
			cursor = hdf.root.task[start_time-(500.*(60./1000)):start_time+(1500.*(60./1000))]['cursor'][:,[0,2]]
		
		curs_b = np.arange(-500,1500,1000./60.)
		vel = np.diff(cursor,axis=0)/(1000/60.)
		filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=0)
		vel_bins = curs_b[:-1] + 0.5*(curs_b[1] - curs_b[0])

		#Target location: 
		st_targ_tm = msg_time[ix+4]
		targ = hdf.root.task[st_targ_tm+1]['target'][[0,2]]
		assert not np.all(targ==0.) #Target is not at the origin
		mc_vect = targ/np.linalg.norm(targ)
		mc_vect_mat = np.tile(mc_vect[np.newaxis,:], (filt_vel.shape[0], 1))

		#Now get kin_sig, kin_feat
		KIN_SIG = np.sum(np.multiply(mc_vect_mat, filt_vel), axis=1)
		start_bin = int(np.argmin(np.abs(curs_b)) - 300*(60/1000.)) #start bin is 300 ms before go cue
	

		# kinsig is -.7:1.3
		# start_bin is -500 (essentially)
		#vel_bins are -500:1500 (really -700:1300)
		kin_feat = pm.get_kin_sig_shenoy(KIN_SIG[np.newaxis], bins=vel_bins, start_bin=start_bin,
			first_local_max_method=True) #returns signal in terms of bins

		rxn_time = kin_feat[0,2]

		kinrow['cursor_traj'] = cursor
		kinrow['mc_vect'] = mc_vect
		kinrow['kin_sig'] = KIN_SIG
		kinrow['kin_feat'] = kin_feat[0,:]
		
		#Get reach error using 'cursor' array: 
		reach_traj = hdf.root.task[go_time:rxn_time]['cursor'][:,[0,2]]
		
		#go_time to either hold penalty or hold w/in reward
		cursor_trunc = hdf.root.task[go_time:msg_time[ix+6]] 
		eps = 10**-10
		rch_error = 0
		orig = hdf.root.task[msg_time[ix+1]]['target'][[0,2]]
		opt_traj_vect = targ - orig + eps
		for pt in range(len(cursor_trunc)):
			trav_traj_vect = pt - orig + eps
			cos_err_ang = np.dot(opt_traj_vect, trav_traj_vect)/\
				(np.linalg.norm(opt_traj_vect)*np.linalg.norm(trav_traj_vect))
			err_ang = np.arccos(cos_err_ang)
			rch_error = rch_error + np.linalg.norm(trav_traj_vect)*np.sin(err_ang)

	else:
		rxn_time = -1
		rch_error = -1
		targ = np.array([-1, -1])

	trial['rxn_time'] = rxn_time
	trial['reach_err'] = rch_error
	trial['target_loc'] = targ

	#Hold Time: 
	if trial_outcome not in ['center_timeout']:
		hold_time = msg_time[ix+3] - msg_time[ix+2] 
		hold_time = hold_time*(1000./60.)
	else:
		hold_time = -1

	trial['hold_time'] = hold_time

	#Reward Rate: 
	start_time = msg_time[ix]
	end_session_time = msg_time[-1]
	if start_time < 0.5*(60*60):
		rews = np.nonzero(rew_times<start_time+(0.5*(60*60)))[0]
		rew_rate = len(rews)/float(0.5+ start_time/(60*60.))
	elif start_time + 0.5*(60*60) > end_session_time:
		rews = np.nonzero(rew_times>start_time-(0.5*(60*60.)))[0]
		rew_rate = len(rews)/float(0.5+ (end_session_time - start_time)/(60*60))
	else:
		rix, rval = fcns.get_in_range(rew_times, [start_time - (0.5*(60.*60)), start_time + (0.5*(60.*60))])
		rew_rate = len(rix)
	trial['rew_rate'] = rew_rate

	return trial, kinrow, TE_go_inds

def get_mc_mem_metrics(start_ind, msg, msg_time, trial, kinrow, TE_go_inds, rew_times, hdf,tsk, te):
	trial_ind = start_ind[0]
	ix = start_ind[1]
	if msg[ix+2]== 'timeout_penalty':
		trial_outcome = 'center_timeout'
	elif ((msg[ix+3]== 'hold_penalty') or (msg[ix+4]== 'hold_penalty') or (msg[ix+5]== 'hold_penalty')):
		trial_outcome = 'center_holderr'
	elif msg[ix+6]== 'wrong_target_penalty':
		trial_outcome = 'wrong_target'
	elif msg[ix+6]== 'timeout_penalty':
		trial_outcome = 'choice_timeout'
	elif msg[ix+7]== 'hold_penalty':
		trial_outcome = 'choice_holderr'
	elif msg[ix+7]== 'reward':
		trial_outcome = 'reward'
	else:
		print 'error: unrecognized trial type, msg index: ', ix
	trial['trial_outcome'] = trial_outcome

	#GO time: 
	if trial_outcome not in ['center_timeout', 'center_holderr']:
		#Get go time: 
		go_time = msg_time[ix+5]
		TE_go_inds[tsk,te].append(go_time)
	else:
		go_time = -1
		TE_go_inds[tsk,te].append(0)

	trial['go_time'] = go_time

	#Reach time: 
	if trial_outcome in ['reward','choice_holderr','wrong_target']:
		assert (msg[ix+5] == 'choice_targets')
		reach_time = (msg_time[ix+6] - msg_time[ix+5])*(1000./60.)
	else:
		reach_time = -1
	trial['reach_time'] = reach_time

	#Get Rxn time and reach error: 
	if trial_outcome in ['reward','choice_holderr','wrong_target']:
		start_time = go_time - 200.*(60./1000)

		if start_time < int(500.*(60./1000)):
			foo = np.arange(-500., 1500., 1000/60.)
			cursor = np.zeros((len(foo), 2))
			cursor_trunc = hdf.root.task[:start_time+(1500.*(60./1000))]['cursor'][:,[0,2]]
			cursor[len(cursor)-cursor_trunc.shape[0]:,:] = cursor_trunc
		
		elif start_time + (1500.*(60./1000)) > hdf.root.task[:]['cursor'].shape[0]:
			foo = np.arange(-500., 1500., 1000/60.)
			cursor = np.zeros((len(foo), 2))
			cursor_trunc = hdf.root.task[start_time-(500.*(60./1000)):]['cursor'][:,[0,2]]
			cursor[:cursor_trunc.shape[0],:] = cursor_trunc
		
		else:
			cursor = hdf.root.task[start_time-(500.*(60./1000)):start_time+(1500.*(60./1000))]['cursor'][:,[0,2]]
		
		curs_b = np.arange(-500,1500,1000./60.)
		vel = np.diff(cursor,axis=0)/(1000/60.)
		filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=0)
		vel_bins = curs_b[:-1] + 0.5*(curs_b[1] - curs_b[0])

		#Target location: 
		st_targ_tm = msg_time[ix+3] #During target flash
		targ = hdf.root.task[st_targ_tm+1]['target'][[0,2]]

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
		
		#Get reach error using 'cursor' array: 
		reach_traj = hdf.root.task[go_time:rxn_time]['cursor'][:,[0,2]]
		
		#go_time to either hold penalty or hold w/in reward
		cursor_trunc = hdf.root.task[go_time:msg_time[ix+6]] 
		eps = 10**-10
		rch_error = 0
		orig = hdf.root.task[msg_time[ix+1]]['target'][[0,2]]
		opt_traj_vect = targ - orig + eps
		for pt in range(len(cursor_trunc)):
			trav_traj_vect = pt - orig + eps
			cos_err_ang = np.dot(opt_traj_vect, trav_traj_vect)/\
				(np.linalg.norm(opt_traj_vect)*np.linalg.norm(trav_traj_vect))
			err_ang = np.arccos(cos_err_ang)
			rch_error = rch_error + np.linalg.norm(trav_traj_vect)*np.sin(err_ang)

	else:
		rxn_time = -1
		rch_error = -1
		targ = np.array([-1, -1])

	trial['rxn_time'] = rxn_time
	trial['reach_err'] = rch_error
	trial['target_loc'] = targ

	#Hold Time #2: 
	if trial_outcome not in ['center_timeout','center_holderr']:
		hold_time = msg_time[ix+5] - msg_time[ix+4] 
		hold_time = hold_time*(1000./60.)
	else:
		hold_time = -1

	trial['hold_time'] = hold_time

	#Reward Rate: 
	start_time = msg_time[ix]
	end_session_time = msg_time[-1]
	if start_time < 0.5*(60*60):
		rews = np.nonzero(rew_times<start_time+(0.5*(60*60)))[0]
		rew_rate = len(rews)/float(0.5+ start_time/(60*60.))
	elif start_time + 0.5*(60*60) > end_session_time:
		rews = np.nonzero(rew_times>start_time-(0.5*(60*60.)))[0]
		rew_rate = len(rews)/float(0.5+ (end_session_time - start_time)/(60*60))
	else:
		rix, rval = fcns.get_in_range(rew_times, [start_time - (0.5*(60.*60)), start_time + (0.5*(60.*60))])
		rew_rate = len(rix)
	trial['rew_rate'] = rew_rate

	return trial, kinrow, TE_go_inds




