# For each trial I want 
# 	0. Type of trial (fin, cue_on, cue_med)
# 	1. LFP Target
# 	2. MC Target
# 	3. RT1
# 	4. RT2
# 	5. LFP Power and phase in 25-40 Hz at GO cue onwards (-1 sec, 1 sec)

# 	In this file, I have a method to extract the LFP signal for each trial, 
#	and then a method to plot the mean spectrograms for each set of dates/blocks

# 	Spectrograms are generated from simple_spec, which takes 2d data arrays as inputs
#	(This is in contrast to basic_spec which takes target-separated dictionaries as inputs
# 		-see parse.py: epoch class and /utils/main.py for examples )

import numpy as np
import scipy.io as sio
import psycho_metrics as pm
import simple_spec as ss
import matplotlib.pyplot as plt
import kinarm
import gc
from tables import *
import datetime

#Plot By LFP target
class Spike_Table(IsDescription):
	trial_num = Float64Col()
	day_i = Float64Col()
	block_i = Float64Col()
	unit =  StringCol(10)

def plot_spec_by_LFP_target(dates,blocks,fname_sequRTs, key,method='MTM',moving_win='standard', 
	from_file=False,chans=[70],**kwargs):
	
	if key is None:
		pre = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'
	else:
		pre = key.pre
	print pre

	fname = pre+fname_sequRTs
	dat = sio.loadmat(fname)
	go_cue_sequ = dat['abs_time_go'][0,:]
	n = dat['n'][0,:]

	if from_file:
		if method is 'MTM':
			dat_spec = sio.loadmat(fname[:-4]+'_spec.mat')

		elif method is 'DFT':
			dat_spec = sio.loadmat(fname[:-4]+'_specDFT.mat')

		signal = dat_spec['signal']
		S = dat_spec['spec']
		f = dat_spec['freq']
		t = dat_spec['bins']
		
	else:
		signal=dict()
		for c,ch in enumerate(chans):
			ch_key = 'AD'+str(ch)
			print kwargs

			if method == 'spikes':
				kwargs['spk_tdy_str'] = datetime.date.today().isoformat()
				f_fail = 0
				while f_fail<10:
					try:
						kwargs['spk_h5file'] = openFile(pre + kwargs['spk_tdy_str'] + fname_sequRTs+ str(f_fail)+'_spikes.h5', mode="w", title='Spike_Rates')
						f_fail = 11
					except:
						f_fail+=1

				kwargs['spk_table'] = kwargs['spk_h5file'].createTable("/", 'spk', Spike_Table, "Spike Rates")
				kwargs['spk_vl_arr'] = kwargs['spk_h5file'].createVLArray("/", 'spk_tm', Float64Atom())
				kwargs['pre'] = pre
				signal, empty = get_sig(dates, blocks, go_cue_sequ, n, signal_type = 'spikes', **kwargs)
				t_ax = np.squeeze(kwargs['t_ax'])
				f_ax = np.squeeze(kwargs['f_ax'])
				tot_trials = len(go_cue_sequ)

			else:
				signal[ch_key], empty = get_sig(dates,blocks,go_cue_sequ,n,channel=ch_key,**kwargs)


			if moving_win is not 'standard':
				moving_window = moving_win
			else:
				if method =='MTM':
					moving_window = [0.2, 0.050]
				elif method == 'DFT':
					moving_window = [0.201, 0.051]

			if method=='MTM':
				Smtm, f, t = ss.MTM_specgram(signal[ch_key].T,movingwin=moving_window)
				if 'SMTM' in locals():
					SMTM = np.vstack((SMTM, np.array([Smtm]) ))
				else:
					SMTM = np.array([Smtm])

			elif method == 'DFT':
				Sdft, f, t = ss.DFT_PSD(signal[ch_key].T,movingwin=moving_window)
				if 'SDFT' in locals():
					SDFT = np.vstack((SDFT, np.array([Sdft]) ))
				else:
					SDFT = np.array([Sdft])

			elif method == 'spikes':
				#ST, UN = ss.spike_rates(signal, t_ax, f_ax, tot_trials, fname_sequRTs)
				print 'done with spikes'

			elif method == 'hilbert':
				f_bands = [[1,15],[15,30],[30,45],[45,58],[62,90],[90,120],[70,120],[25,40],[1,120]]
				# A, P are freq x trials x time
				A = np.zeros(( len(f_bands),signal[ch_key].shape[0], signal[ch_key].shape[1] ))
				P = np.zeros(( len(f_bands),signal[ch_key].shape[0], signal[ch_key].shape[1] ))
				for f,fb in enumerate(f_bands):
					amp, phase = ss.hilbert_transform(signal[ch_key].T,bandpass=fb,Fs=1000)
					A[f,:,:] = amp.T
					P[f,:,:] = phase.T

				if 'AHIL' in locals():
					AHIL = np.vstack((AHIL, np.array([A]) ))
					PHIL = np.vstack((PHIL, np.array([P]) ))
				else:
					AHIL = np.array([A])
					PHIL = np.array([P])

	if method != 'spikes':
		signal['chan_ord'] = chans

	if method =='hilbert':
		return AHIL, PHIL, f_bands, signal
	
	elif method =='MTM':
		return SMTM, f, t, signal

	elif method =='DFT':
		return SDFT, f, t, signal

	#elif method == 'spikes':

		#return ST, f_ax, t_ax, signal, UN

		# fig, ax = plt.subplots(nrows=4, ncols=1)
		# for l,lfp in enumerate(key.lfp_targ):
		# 	ind_lfp = np.nonzero(dat['LFP_label'][0,:] == lfp)[0]
		# 	x = np.mean(S[ind_lfp],axis=0)
		# 	#im = ax[l].pcolormesh(t-1.5,f[f<100],x.T,vmin=0, vmax=0.03)
		# 	im = ax[l].pcolormesh(t-1.5,f[f<100],x.T,vmin=0,vmax=1)
		# 	ax[l].axis('tight')
		# 	ax[l].set_title('LFP Target: '+str(l))
		# 	ax[l].set_xlabel('Time (sec)')
		# 	ax[l].set_ylabel('Freq (Hz)')
		# 	ax[l].plot([1.5, 1.5], [0, 100], 'k-')
		# 	ax[l].plot([-1.5, 1.5], [25, 25], 'k--')
		# 	ax[l].plot([-1.5, 1.5], [40, 40], 'k--')
		#cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
		#fig.colorbar(im, cax=cax)
		#fig.tight_layout()
		
def get_sig(dates,blocks,go_cue_sequ,N,
	signal_type='neural',channel='AD70',
	before_go=1.5, after_go=1.5,mc_lab=None,prep=False,**kwargs):

	if 'anim' in kwargs.keys():
		anim = kwargs['anim']
	else:
		anim = 'seba'

	x = np.arange(-1*before_go,after_go,0.001)
	if signal_type is 'endpt':
		signal = np.zeros((go_cue_sequ.shape[0],len(x), 2))
	
	elif signal_type is 'spikes':
		signal = dict()
			
	else:
		signal = np.zeros((go_cue_sequ.shape[0],len(x)))

	if 'all_kins' in kwargs.keys():
		#Xpos, Ypos, XVel, YVel, Speed
		signal2 = np.zeros((5, go_cue_sequ.shape[0],len(x)))

	targ_dir_array = np.zeros((go_cue_sequ.shape[0], 2))
	ind_prev = 0
	cnt = 0
	d = 0
	b = -1

	if 'target_coding' in kwargs.keys():
		target_coding = kwargs['target_coding']
	else:
		target_coding = 'none'

	if signal_type == 'shenoy_jt_vel' or signal_type == 'endpt':
		params = sio.loadmat('/Users/preeyakhanna/Dropbox/Carmena_Lab/lfp_multitask/analysis/KinematicsParameters_seba.mat')
		calib = kinarm.calib(params['sho_pos_x'][0,0], params['sho_pos_y'][0,0], params['L1'][0,0], params['L2'][0,0], params['L2ptr'][0,0])

		if signal_type == 'shenoy_jt_vel':
			targ_dir = dict()
			if np.any(mc_lab.astype(int)==70) or prep==True or target_coding=='standard':
				targ_dir[64] = np.array([1, 0])
				targ_dir[65] = np.array([1, 1])/np.sqrt(2)
				targ_dir[66] = np.array([0, 1])
				targ_dir[67] = np.array([-1, 1])/np.sqrt(2)
				targ_dir[68] = np.array([-1, 0])
				targ_dir[69] = np.array([-1, -1])/np.sqrt(2)
				targ_dir[70] = np.array([0, -1])
				targ_dir[71] = np.array([1, -1])/np.sqrt(2)

				targ_dir[72] = targ_dir[64].copy()
				targ_dir[73] = targ_dir[65].copy()
				targ_dir[74] = targ_dir[66].copy()
				targ_dir[75] = targ_dir[67].copy()
				targ_dir[76] = targ_dir[68].copy()
				targ_dir[77] = targ_dir[69].copy()

				#Repeated trials
				targ_dir[164] = np.array([1, 0])
				targ_dir[165] = np.array([1, 1])/np.sqrt(2)
				targ_dir[166] = np.array([0, 1])
				targ_dir[167] = np.array([-1, 1])/np.sqrt(2)
				targ_dir[168] = np.array([-1, 0])
				targ_dir[169] = np.array([-1, -1])/np.sqrt(2)
				targ_dir[170] = np.array([0, -1])
				targ_dir[171] = np.array([1, -1])/np.sqrt(2)

			else:
				print 'alert: alternative coding'
				print moose
				targ_dir[64] = np.array([1, 0])
				targ_dir[65] = np.array([0, 1])
				targ_dir[66] = np.array([-1, 0])
				targ_dir[67] = np.array([0, -1])


	for j,n in enumerate(N):
		print 'NEW BLOCK: '+str(j)+' '+str(n)
		#Increment b, d:
		ind_prev = 0
		b+=1
		if b >= len(blocks[d][0]):
			b = 0 
			d+=1

		#If trials: 
		if n > 0:
			index = np.arange(cnt,cnt+n)
			
			if signal_type is 'neural':
				fname = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'+anim+dates[d]+blocks[d][0][b]+'.mat'
				dat = sio.loadmat(fname,variable_names=channel)
				dat = dat[channel]
			
			elif signal_type is 'jt_vel':
				fname = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'+anim+dates[d]+blocks[d][0][b]+'_kin.mat'
				datx = sio.loadmat(fname,variable_names='AD35')
				datx = datx['AD35']
				daty = sio.loadmat(fname,variable_names='AD36')
				daty = daty['AD36']
				dat = np.sqrt(datx**2 + daty**2)

			elif signal_type is 'shenoy_jt_vel' or signal_type is 'endpt':
				mc_lab_ind = mc_lab[index]
				fname = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'+anim+dates[d]+blocks[d][0][b]+'_kin.mat'
				x = sio.loadmat(fname)
				dat = np.vstack((x['AD33'], x['AD34'], x['AD35'], x['AD36'])).T

			elif signal_type is 'spikes':
				fname = kwargs['pre']+anim+dates[d]+blocks[d][0][b]+'.mat'
				dat_dict = sio.loadmat(fname)
				unit_dict = [key for k, key in enumerate(dat_dict.keys()) if key[0:3] == 'sig']
				ad_dict = [key for k, key in enumerate(dat_dict.keys()) if key[0:2] == 'AD']
				x = dat_dict[ad_dict[0]].T
				dat = np.zeros((x.shape[0], 1))
			
			if dat.shape[0]<dat.shape[1]:
				dat = dat.T
		
			for t,ind in enumerate(go_cue_sequ[index]):

				if ind < ind_prev: 
					print 'error! ahhhhh!'
					print ind, ind_prev

				#get desired signal
				if ind<1.5:
					try:
						tmp = range(int(1000.*ind)+1500)
						signal[t+cnt,-len(tmp):] = dat[tmp,0]
					except:
						print 'non trial'
					print 'exception1, ' +str(t)

				elif ind>((dat.shape[0]/1000.)-1.5):
					try:
						tmp = np.arange(int(1000.*ind)-1500,dat.shape[0])
						signal[t+cnt,0:len(tmp)] = dat[tmp,0]
					except:
						print 'non trial'
					print 'exception2, '+str(t)+'   ' + str(dat.shape[0]/1000.)+ '     '+str(ind_prev)+'   '+str(ind)
				
				else:
					if signal_type is 'shenoy_jt_vel' or signal_type is 'endpt':
						
						sig = dat[int(1000.*ind)-int(before_go*1000):int(1000.*ind)+int(after_go*1000),:]
						L = int(after_go*1000) + int(before_go*1000)
						
						#Convert to X, Y space
						xpos,ypos,xvel,yvel = kinarm.calc_endpt(calib,sig[:,0], sig[:,1], sh_vel = sig[:,2], el_vel = sig[:,3])
						
						if signal_type is 'shenoy_jt_vel':
							kin = np.hstack(( np.array([xvel]).T, np.array([yvel]).T ))
							#try:
							try:
								vec = np.tile(np.array([targ_dir[int(mc_lab_ind[t])]]).T, (1,L))
								signal[t+cnt,:] = np.diag(np.matrix( kin )*np.matrix( vec ))
								targ_dir_array[t+cnt,:] = targ_dir[int(mc_lab_ind[t])]

								if 'all_kins' in kwargs.keys():
									signal2[0, t+cnt, :] = xpos
									signal2[1, t+cnt, :] = ypos
									signal2[2, t+cnt, :] = xvel
									signal2[3, t+cnt, :] = yvel
									signal2[4, t+cnt, :] = np.sqrt(xvel**2 + yvel**2)
							
							except:
								'print cant do for trial'+str(ind)
								#Repeat trials
							#	vec = np.tile(np.array([targ_dir[int(mc_lab_ind[t]-100)]]).T, (1,L))
						elif signal_type is 'endpt':
							signal[t+cnt,:,0] = xpos
							signal[t+cnt,:,1] = ypos


					elif signal_type is 'jt_vel' or signal_type is 'neural':
						signal[t+cnt,:] = dat[int(1000.*ind)-int(1000*before_go):int(1000.*ind)+int(1000*after_go),0]

					elif signal_type is 'spikes':
						for u in unit_dict:
							#New row for each trial 
							trl = kwargs['spk_table'].row
							trl['trial_num'] = t+cnt
							trl['day_i'] = d
							trl['block_i'] = b
							trl['unit'] = u


							spk_ix = np.nonzero(np.logical_and(dat_dict[u]>=(ind-before_go), dat_dict[u]<(ind+after_go)))[1]
							spk_tm = dat_dict[u][0, spk_ix] - ind + before_go
							kwargs['spk_vl_arr'].append(spk_tm)


							#tm_edges = np.arange(0, before_go + after_go, 1./1000.)
							#S = np.zeros((len(tm_edges), ))
							#S[(spk_tm*1000).astype(int)]= 1;


							# def tm_to_rt(spk_tm, before_go, after_go):
							# 	tm_edges = np.arange(-1*before_go, after_go+(1./1000.), 1./1000.)
							# 	S = np.zeros((len(tm_edges)-1, ))
							# 	for i_t, tm in enumerate(tm_edges[:-1]):
							# 		ix = np.nonzero(np.logical_and(spk_tm<=tm_edges[i_t+1], spk_tm>tm))[0]
							# 		S[i_t] = len(ix)
							# 	return S

							#trl['spikes'] = S
							trl.append()
						kwargs['spk_table'].flush()

				#print t+cnt
				ind_prev = ind.copy()
		cnt = cnt + n


	#print 'Final : day '+str(d), 'block '+str(b), 'N '+str(n), 'Total Cnt '+str(cnt)
	if signal_type is 'shenoy_jt_vel':
		if 'all_kins' in kwargs.keys():
			return signal, signal2, targ_dir_array
		else:
			return signal, targ_dir_array
	elif signal_type is 'spikes':
		kwargs['spk_h5file'].close()
		return True, 0
	else:
		return signal, 0


