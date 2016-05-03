
from tables import *
import numpy as np
import basic_spec
import signal_utils_Morton as su

def make(e,data,fname):
	h5file = open_file(fname, mode = "w", title = "Trial_stack for "+fname)
	data_table = h5file.create_table("/", 'trial_data', Trial_Data,"Info for Trial")
	t_data = data_table.row 

	#Array for neural data: 
	n_data = h5file.create_group("/","n_data","Neural Data")

	#Array for spectral data: 
	s_data = h5file.create_group("/","s_data","Spectral Data")	

	strobed = data['Strobed']
	start_trials = np.nonzero(strobed[:,1]==e.codes['cue_on'])[0]
	start_times = strobed[start_trials,0]
	time_array = np.arange(-1,2,0.001)

	# possible trial ends: 
	trial_end = np.array([e.codes['wrong_target'], e.codes['ph_error'], e.codes['timeout_error'], e.codes['reward_on']])
	
	#Adding factor to targets 
	targ_ind_add = e.codes['strobed_targ_ind_add'] = 64

	#Time series holder
	n_dat_holder = np.zeros((4, len(start_trials), len(time_array)))
	
	#Args for specgram
	kwar = basic_spec.generate_SU_params()

	for t,tind in enumerate(start_trials):

		Pxx_list = []
		post_start = strobed[tind:,1]

		#Look for next 'trial_ender':
		ind = np.zeros((len(trial_end),1))+strobed.shape[0]+1
		for i,code in enumerate(trial_end):
			tmp = np.nonzero(post_start==code)[0]

			if len(tmp)>0:
				ind[i] = tmp[0]

		if np.any(ind<strobed.shape[0]+1):
			trial_code = strobed[int(np.min(ind))+tind,1]
			RT = strobed[int(np.min(ind))+tind,0] - strobed[tind,0]
			target_ind = np.nonzero(strobed[tind:int(np.min(ind))+tind,1]>= targ_ind_add)[0]
			target = int(strobed[tind+target_ind,1][0]) - targ_ind_add

			t_data['trial_index'] = t
			t_data['trial_code'] = trial_code
			t_data['reaction_time'] = RT
			t_data['target'] = target
			t_data.append()

			times = np.round(start_times[t]*e.file_feats['fs'])+(time_array*e.file_feats['fs']); 
			times[times<0] = 0
			times[times>data['AD65'].shape[1]] = data['AD65'].shape[1]

			n_dat_holder[0,t,:] = data['AD70'][0,times.astype(int)]
			n_dat_holder[1,t,:] = data['AD100'][0,times.astype(int)]
			n_dat_holder[2,t,:] = data['AD150'][0,times.astype(int)]
			n_dat_holder[3,t,:] = data['AD170'][0,times.astype(int)]	

			for c in range(4):
				Pxx, freqs, bins = su.specgram(n_dat_holder[c,t,:], **kwar)
				Pxx_list.append(Pxx)

			Pxx_array = np.array(Pxx_list)
			Pxx_array = np.reshape(Pxx_array, (1, Pxx_array.shape[0], freqs.shape[0], bins.shape[0]))

			if 'Pxx_stack' not in locals():
				Pxx_stack = Pxx_array.copy()
			else:
				Pxx_stack = np.vstack((Pxx_stack,Pxx_array))
			del Pxx_list


	a = np.sum(n_dat_holder,axis=2)
	non_trials = np.nonzero(a==0)[0]
	n_dat_holder = np.delete(n_dat_holder, (non_trials), axis = 1)
	h5file.createArray(n_data, 'trials', n_dat_holder,"chan_trial_time")

	h5file.createArray(s_data,'specgrams',Pxx_stack," ")

	h5file.close()

class Trial_Data(IsDescription):
	trial_index = Int32Col()
	trial_code = Int32Col()
	reaction_time = Float32Col()
	target = Int32Col()

