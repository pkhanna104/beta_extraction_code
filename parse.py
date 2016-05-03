import numpy as np
import scipy
import scipy.io as sio
import sys
import os.path

try:
	import pre_process_LFP as ppL
	import plx_search
	import make_trial_table
	import kinematics
except:
	'plx_search, make_trial_table, and kinematics not available'


# fin_dates = ['050314','050414','050514','050614','050714','050814'] 
# fin_blocks=[['efg'],['abdefg'],['abcdefghij'],['abcdefghijkl'],['abcdefghij'],['abcdefghi']]

# e = parse.epoch(5,dates=fin_dates, blocks=fin_blocks)
# e.mat_file_check()


class epoch:
	def __init__(self, align_code=5, dates=['032014','032514'], blocks=[['abc'],['abcde']], fs=1000, 
		before_code=1, after_code=1, only_rew_trials=1, notch_filter=True, kin_only=False, 
		chans = range(65,193),anim='seba',folder_name = 'seba'):
		file_feats = dict()
		file_feats['anim'] = anim
		file_feats['dates'] = dates #list of dates: format is list of strings
		file_feats['blocks'] = blocks #format is list of list of string with all blocks: [['abc'],['bcde']]
		file_feats['kin_only'] = kin_only
		file_feats['backup_folder_name'] = folder_name

		if kin_only:
			file_feats['AD_chan'] = range(33,37)
		else:
			file_feats['AD_chan'] = chans

		file_feats['fs'] = fs
		file_feats['notch_filter'] = notch_filter
		file_feats['mat_data_path'] = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'
		#file_feats['mat_data_path'] = '/work/pkhanna/spec_python/mat_files/'
		file_feats['plx_data_path'] = '/Volumes/carmena/' + file_feats['backup_folder_name'] + '/'

		codes = dict() #From XML Files
		codes['go'] = 5
		codes['cue_on'] = 2
		codes['leave_orig'] = 6
		codes['enter_target'] = 7
		codes['reward_on'] = 9
		codes['end_trial'] = 11
		codes['strobed_targ_ind_add'] = 64
		

		codes['wrong_target'] = 20

		#TODO: update these: 
		codes['ph_error'] = 8
		codes['timeout_error'] = 12

		epoch_feats = dict()
		epoch_feats['align_code'] = align_code
		epoch_feats['before_code'] = before_code
		epoch_feats['after_code'] = after_code
		epoch_feats['only_rew_trials'] = only_rew_trials

		self.file_feats = file_feats
		self.codes = codes
		self.epoch_feats = epoch_feats

	def mat_file_check(self,override=False,Strobed_andAD70=False):
		for d,day in enumerate(self.file_feats['dates']):
			for b,block in enumerate(self.file_feats['blocks'][d][0]):

				if self.file_feats['kin_only']:
					mat_filename = self.file_feats['mat_data_path'] + self.file_feats['anim'] + day + block + '_kin.mat'
				else:
					mat_filename = self.file_feats['mat_data_path'] + self.file_feats['anim'] + day + block + '.mat'

				plx_pre2 = self.file_feats['plx_data_path'] + self.file_feats['anim'] + day + '/'
				plx_pre1 = self.file_feats['plx_data_path'] + self.file_feats['anim'] + day + '/map_data/'

				is_file = os.path.isfile(mat_filename)
				if is_file:
					print mat_filename
					m = sio.loadmat(mat_filename)
					ch_list = [int(ix[2:]) for j,ix in enumerate(m.keys()) if ix[:2] in ['AD']]
					missing_ch = [ch for j, ch in enumerate(self.file_feats['AD_chan']) if ch not in ch_list]
					if len(missing_ch)==0:
						all_chan = True
					else:
						all_chan = False

				if (not is_file) or override or (not all_chan): #make file again if 1. no file, 2. override, or 3. missing channels
					print 'making .mat for ' + day + block
					if Strobed_andAD70:
						chan = [70]
					else:
						chan = self.file_feats['AD_chan']
					print chan
					
					try:
						plx_search.make_mat(plx_pre1+self.file_feats['anim']+day+block+'.plx', mat_filename, channels=chan)
				
					except:
					#	print chan
						plx_search.make_mat(plx_pre2+self.file_feats['anim']+day+block+'.plx', mat_filename, channels=chan)
				

	
	def add_channels(self):
		for d,day in enumerate(self.file_feats['dates']):
			for b,block in enumerate(self.file_feats['blocks'][d][0]):
				
				mat_filename = self.file_feats['mat_data_path'] + self.file_feats['anim'] + day + block + '.mat'
				#plx_pre2 = self.file_feats['plx_data_path'] + self.file_feats['anim'] + day + '/'
				plx_pre1 = self.file_feats['plx_data_path'] + self.file_feats['anim'] + day + '/map_data/'

				chan = list(np.arange(70,96,2)) + list(np.arange(129,128+32))
				if os.path.isfile(mat_filename):
					x = sio.loadmat(mat_filename)
					if 'AD74' not in x.keys():
						plx_search.make_mat(plx_pre1+self.file_feats['anim']+day+block+'.plx', mat_filename, channels=chan)
				else:
					plx_search.make_mat(plx_pre1+self.file_feats['anim']+day+block+'.plx', mat_filename, channels=chan)

					
	def parse_all(self,channel_list,parse_type='reward'):
		#Initializing 'trials' ndarray
		#First define t
		t = np.arange(-1*self.epoch_feats['before_code'], self.epoch_feats['after_code'], float(1)/float(self.file_feats['fs']))

		#Then find maximum block number
		m =1;
		for b,bl in enumerate(self.file_feats['blocks']):
			if len(bl[0])>m:
				m=len(bl[0])

		#Initialize trial
		trials = np.empty((8, len(self.file_feats['dates']), m , len(channel_list), 500, len(t))) #targets x days x blocks x channel x trials

		#Loop through each file
		for d,day in enumerate(self.file_feats['dates']):
			for b,block in enumerate(self.file_feats['blocks'][d][0]):

				#Load stored .mat file (generated from plx_search.sh, plx_to_mat.py)
				print 'loading day: ' + str(day) + ', block: ' + str(block)
				fname = self.file_feats['mat_data_path'] + self.file_feats['anim'] + day + block + '.mat'
				data = sio.loadmat(fname)
				print 'loaded data, extracting trials'
				data['day'] = day
				data['block'] = block

				if parse_type == 'all':
					make_trial_table.make(self,data,self.file_feats['anim'] + day + block)

				else:
					#Loop through each target
					for t in range(0,8):

						#Call parse_ind: 
						if parse_type=='reward':
							num_tri, tri = self.parse_ind(data, t,1, self.codes['end_trial'],channel_list)

						elif parse_type == 'wrong_target':
							num_tri, tri, err_ind = self.parse_ind(data, t, 0, self.codes['wrong_target'])

						if num_tri > 0: 
							trials[ t, d, b, :, :num_tri, :] = tri 
							if parse_type == 'wrong_target':
								err[t] = error_targ

		#Store all trials in self.trials
		if parse_type == 'reward':
			self.trials = trials

		elif parse_type == 'wrong_target':
			self.wrong_targ_trials = trials
			targ_est = kinematics.get_target_est(data, error_ind)
			self.wrong_targ_trials_err = targ_est

	def parse_ind(self, data, target, reward_only, trial_ender,channel_list):
		# Returns a channel x trials x time ndarray for each target in a plx file
		try:
			strobed = data['Strobed']
		except:
			print 'redoing .mat file'

			#Test if server is connected
			try:
				test = sio.loadmat('/Volumes/carmena/seba/seba052713/bmi_data/dec052713a_bmi.mat')
			except:
				print "Labshare not found! Ah!"
				resp = raw_input("Please enter 'y' when you have loaded the labshare: ")
				
				while resp is not 'y':
					resp = raw_input("Please enter 'y' when you have loaded the labshare: ")
				
				try:
					test = sio.loadmat('/Volumes/carmena/seba/seba052713/bmi_data/dec052713a_bmi.mat')
				except:
					print 'loading failed :/'
				
			try:
				plx_search.make_mat(self.file_feats['plx_data_path'] + self.file_feats['anim'] + data['day'] + '/', data['block'], self.file_feats['mat_data_path'],self.file_feats['anim'] + data['day'])
			except:
				plx_search.make_mat(self.file_feats['plx_data_path'] + self.file_feats['anim'] + data['day'] + '/map_data/', data['block'], self.file_feats['mat_data_path'],self.file_feats['anim'] + data['day'])
			
			fname = self.file_feats['mat_data_path'] + self.file_feats['anim'] +  data['day'] +  data['block'] + '.mat'
			data = sio.loadmat(fname)
			strobed = data['Strobed']

		target = target + self.codes['strobed_targ_ind_add']

		#Get times for [pre, post] code
		start_trials = np.nonzero(strobed[:,1]==self.codes['cue_on'])
		full_trials = np.vstack((start_trials[0],np.zeros(start_trials[0].shape))).T
		end_trials = np.nonzero(strobed[:,1]==trial_ender)
		full_trials=full_trials.astype(int)
		
		#For each start_trials index, find appropriate end_trials index. Store in full_trials
		for t,tind in enumerate(full_trials[:,0]):
			t1 = tind<end_trials[0]

			try:
				t2 = end_trials[0]<int(full_trials[t+1,0])
			except:
				t2 = t1.copy()
			e = np.nonzero(scipy.logical_and(t1,t2))[0]

			if len(e) == 1: #Add endpoint
				full_trials[t,1] = end_trials[0][e[0]]

			elif len(e) == 0: #Remove trial
				full_trials[t,0] = 0
			else: 
				print ('Found too many trial ends for start index %d ' %t) 
				full_trials[t,0]=0

		#Now full_trials is in Strobed coordinates 
		full_trials = full_trials[np.nonzero(np.sum(full_trials,axis=1))[0],:] 

		#Exclude Fails and get only correct target
		for t,tind in enumerate(full_trials[:,0]):
			tmp = strobed[tind:full_trials[t,1],1]

			#Remove failures (no reward)
			if reward_only:
				if len(np.nonzero(tmp==self.codes['reward_on'])[0]) == 0:
					full_trials[t,:] = 0 

			#Remove trials with wrong target
			if len(np.nonzero(tmp == target)[0]) == 0:
				full_trials[t,:] = 0 

		full_trials = full_trials[np.nonzero(np.sum(full_trials,axis=1))[0],:] #In Strobed Coordinates

		if trial_ender == self.codes['wrong_target']:
			err_ind = strobed[full_trials[:,1],0] #In real time (sec)

		#If trials exist still
		if len(full_trials) > 0:

			#Get align_code time, convert Strobed indices to ADxx indices, store in AD_ind
			AD_ind = np.zeros((full_trials.shape[0],1))

			for t,tind in enumerate(full_trials[:,0]):
				tmp = strobed[tind:full_trials[t,1],1]
				strobed_ind = np.nonzero(tmp == self.epoch_feats['align_code'])[0] + tind
				AD_ind[t] = np.round(strobed[strobed_ind[0],0] * self.file_feats['fs'])
	
			#Now go through each channel and store trial aligned activity
			t = np.arange(-1*self.epoch_feats['before_code'], self.epoch_feats['after_code'], float(1)/float(self.file_feats['fs']))
			start_ind = -1*np.round(self.epoch_feats['before_code']*self.file_feats['fs'])
			end_ind = np.round(self.epoch_feats['after_code']*self.file_feats['fs'])


			#Initialize channel x trial x time ndarray: 
			chan_trial_data = np.empty((len(channel_list), len(AD_ind), len(t)))
			
			#Counter for mixplaced trials
			whoops = 0

			#for c,chan in enumerate(self.file_feats['AD_chan']):
			for c,chan in enumerate(channel_list):
				chan_index = c
				ad_str = 'AD' + str(chan)

				# Preprocess AD file: 
				signal = data[ad_str]

				#Input is a 1xt array: 
				proc_sig = ppL.pre_proc_LFP(signal,method='iir',demean='yes')

				for t,ad_ind in enumerate(AD_ind):
					
					try: #if fits correctly
						chan_trial_data[chan_index,t,:] = proc_sig[start_ind+int(ad_ind):end_ind+int(ad_ind)]
					except: #if at beginning or end of trial
						whoops += 1

					#print 'indices: ',start_ind, end_ind, ad_ind, type(ad_ind)
					# print chan_trial_data.shape
					# print chan_index, t
					# print chan_trial_data[chan_index,t,:].shape
					# print proc_sig[start_ind+int(ad_ind):end_ind+int(ad_ind)].shape
					#print 'chan_trial_data shape: ', chan_trial_data.shape
					#print 'proc_sig[start_ind+ad_ind:end_ind+ad_ind]: ', len(proc_sig[start_ind+ad_ind:end_ind+ad_ind])
					#whoops = message_to_trigger_error

			print 'total misaligned trials: ', whoops
			
			#Return number of trials, ndarray: 
			if trial_ender == self.codes['end_trial']: #Reward trials
				return len(AD_ind), chan_trial_data

			elif trial_ender == self.codes['wrong_target']: #Wrong target trials
				return len(AD_ind), chan_trial_data, err_ind

		else:
			return 0, np.array([0])












