import parse
import generate_spec_metrics as gsm
import psycho_metrics as pm
import scipy.io as sio
import numpy as np
import spectral_metrics as sm

key = pm.codes()
pre = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'

# #dates: beta frac controlled (want targ 2,3,4)
# fin_dates = ['050314','050414','050514','050714','050814'] 
# fin_blocks=[['efg'],['abdefg'],['abcdefghij'],['abcdefghij'],['abcdefghi']]
# #e1 = parse.epoch(5,dates=fin_dates, blocks=fin_blocks, kin_only=True)
# #e1.mat_file_check()
# pm.RT_files(fin_dates,fin_blocks,key,'v1_final_sequRTs.mat',sequential=True)
# gsm.gen_spec_metrics([fin_dates], [fin_blocks], ['v1_final_sequRTs.mat'])
# KIN_FEAT = pm.get_kinematic_features(fin_dates,fin_blocks,key)
# d = dict()
# d['kin_feat'] = KIN_FEAT
# sio.savemat(pre+'v1_final_sequ_kin_mets.mat',d)

# #dates: beta power controlled (want targ 1,2,3)
# d = ['060614','061014','061114']
# b = [['abcdefghijklmn'],['bcdefghijkl'],['cdefgh']]
# pm.RT_files(d,b,key,'v1_final2_sequRTs.mat',sequential=True)
# gsm.gen_spec_metrics([d], [b], ['v1_final2_sequRTs.mat'])
# KIN_FEAT = pm.get_kinematic_features(d,b,key)
# d = dict()
# d['kin_feat'] = KIN_FEAT
# sio.savemat(pre+'v1_final2_sequ_kin_mets.mat',d)

# #cue_dates: beta fraction controlled (want targ 2,3,4)
# cue_on_dates = ['050914','051314','051414','051514','051614']
# cue_on_blocks = [['efgh'], ['abcdefghij'],['abcdefghijk'],['abcdefghi'],['abcdefghi']]
# #e2 = parse.epoch(5,dates=cue_on_dates, blocks=cue_on_blocks)
# pm.RT_files(cue_on_dates,cue_on_blocks,key,'cue_early_sequRTs.mat',sequential=True)
# gsm.gen_spec_metrics([cue_on_dates], [cue_on_blocks], ['cue_early_sequRTs.mat'])
# KIN_FEAT = pm.get_kinematic_features(cue_on_dates,cue_on_blocks,key)
# d = dict()
# d['kin_feat'] = KIN_FEAT
# sio.savemat(pre+'cue_early_sequ_kin_mets.mat',d)

# #cue_dates : beta power controlled (want targ 1,2,3)
# dates = ['060414', '060514']
# blocks = [['abcdefghijk'],['abcdefghijk']]
# #e2 = parse.epoch(5,dates=dates, blocks=blocks)
# pm.RT_files(dates,blocks,key,'cue_early2_sequRTs.mat',sequential=True)
# gsm.gen_spec_metrics([dates], [blocks], ['cue_early2_sequRTs.mat'])
# KIN_FEAT = pm.get_kinematic_features(dates,blocks,key)
# d = dict()
# d['kin_feat'] = KIN_FEAT
# sio.savemat(pre+'cue_early2_sequ_kin_mets.mat',d)

#Create RT,spec,specDFT, kin metric files: 
def create_all(dates, blocks, key, fname='v1_final_sequRTs.mat',task='multi',
	mov_win_specs = [[0.201,0.051], [0.201, 0.011], [0.101, 0.011]],**kwargs):
	'''suitable for tasks: 
		multi
		flash_cue
		MC1, MC2
	'''
	print task
	pm.RT_files(dates, blocks, key, fname, sequential=True,task=task,**kwargs)
	gsm.gen_spec_metrics([dates], [blocks], [fname],mov_win_specs = mov_win_specs,task=task,**kwargs)
	KIN_FEAT, KIN_SIG,v2 = pm.get_kinematic_features(dates, blocks, key,task=task,**kwargs)
	d = dict()
	d['kin_feat'] = KIN_FEAT
	d['kin_sig'] = KIN_SIG
	kin_fname = fname[:-7]+'_kin_mets.mat'
	sio.savemat(pre+'manual_control_redo/'+kin_fname,d)


def create_hilbert(dates,blocks,key,fname='v1_final_sequRTs.mat'):
	A, P, f_bands, signal = sm.plot_spec_by_LFP_target(dates,blocks,fname, key,method='hilbert',from_file=False)
	fname = pre + fname[:-4] + '_spec_hilbert.mat'
	d = dict()
	d['amp'] = A
	d['phase'] = P
	d['f_bands'] = f_bands
	d['signal'] = signal
	sio.savemat(fname,d)

##Concatenate:
def concatenate_all_mets(bf_fname, bp_fname,cat_fname, bf_targ=[85,86,87], bp_targ=[84,85,86]):
	pre = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'
	bf_ind = sio.loadmat(pre+bf_fname)
	beta_frac_ind = np.array([ i for i,t in enumerate(bf_ind['LFP_label'][0,:]) if t in bf_targ])

	bp_ind = sio.loadmat(pre+bp_fname)
	beta_pow_ind = np.array([ i for i,t in enumerate(bp_ind['LFP_label'][0,:]) if t in bp_targ])

	d = dict()
	for k,key in enumerate(bp_ind.keys()):
	    if key[0]=='_' or key=='n' or key=='trial_type':
	        print key+'skip'
	    else:
	        d[key] = np.hstack(( bf_ind[key][0,beta_frac_ind], bp_ind[key][0,beta_pow_ind]))

	n_bf = bf_ind['n'][0,:]
	n_bp = bp_ind['n'][0,:]
	new_n = np.zeros((len(n_bf)+len(n_bp),))
	N=np.hstack((np.array([0]), np.cumsum(n_bf)))
	N=np.hstack((N, N[-1]+np.cumsum(n_bp)))

	ind = np.zeros((len(bf_ind['LFP_label'][0,:])+ len(bp_ind['LFP_label'][0,:]) ))
	ind[beta_frac_ind] = 1
	ind[len(bf_ind['LFP_label'][0,:])+beta_pow_ind] = 1

	#ind2 = np.hstack(( bf_ind['LFP_label'][0,:] >84, bp_ind['LFP_label'][0,:] <87))

	for i in range(len(n_bf)+len(n_bp)):
	    start = N[i]
	    stop = N[i+1]
	    new_n[i] = np.sum(ind[start:stop])

	d['n']=new_n
	sio.savemat(pre+cat_fname+'_sequRTs.mat',d)

	#Number of windows: 0,1,2
	for wind in range(3):
		#DFT Spec
		bf_specDFT = sio.loadmat(pre+bf_fname[:-4]+'_specDFT_win'+str(wind)+'.mat')
		bp_specDFT = sio.loadmat(pre+bp_fname[:-4]+'_specDFT_win'+str(wind)+'.mat')

		d = dict()
		for k,key in enumerate(bf_specDFT.keys()):
		    if key=='signal':
		        d[key] = np.vstack(( bf_specDFT[key][beta_frac_ind,:], bp_specDFT[key][beta_pow_ind, :]))
		    elif key=='spec':
		        d[key] = np.vstack(( bf_specDFT[key][beta_frac_ind,:,:], bp_specDFT[key][beta_pow_ind,:,:]))
		    else:
		        print key+'skip'
		d['freq']= bf_specDFT['freq']
		d['bins']= bf_specDFT['bins']
		sio.savemat(pre+cat_fname+'_sequRTs_specDFT_win'+str(wind)+'.mat',d)

		#MTM Spec
		bf_specDFT = sio.loadmat(pre+bf_fname[:-4]+'_spec_win'+str(wind)+'.mat')
		bp_specDFT = sio.loadmat(pre+bp_fname[:-4]+'_spec_win'+str(wind)+'.mat')

		d = dict()
		for k,key in enumerate(bf_specDFT.keys()):
		    if key=='signal':
		        d[key] = np.vstack(( bf_specDFT[key][beta_frac_ind,:], bp_specDFT[key][beta_pow_ind, :]))
		    elif key=='spec':
		        d[key] = np.vstack(( bf_specDFT[key][beta_frac_ind,:,:], bp_specDFT[key][beta_pow_ind,:,:]))
		    else:
		        print key+'skip'
		d['freq']= bf_specDFT['freq']
		d['bins']= bf_specDFT['bins']
		sio.savemat(pre+cat_fname+'_sequRTs_spec_win'+str(wind)+'.mat',d)


	#KIN
	d_v1 = sio.loadmat(pre+bf_fname[:-7]+'_kin_mets.mat')
	KIN_FEAT_v1 = d_v1['kin_feat']

	d_v2 = sio.loadmat(pre+bp_fname[:-7]+'_kin_mets.mat')
	KIN_FEAT_v2 = d_v2['kin_feat']

	d = dict()
	d['kin_feat'] = np.vstack(( KIN_FEAT_v1[beta_frac_ind], KIN_FEAT_v2[beta_pow_ind]))
	sio.savemat(pre+cat_fname+'_sequ_kin_mets.mat',d)
	
	#hilbert: 
	d_v1 = sio.loadmat(pre+bf_fname[:-4]+'_spec_hilbert.mat')
	A1 = d_v1['amp']
	P1 = d_v1['phase']
	S1 = d_v1['signal']

	d_v2 = sio.loadmat(pre+bp_fname[:-4]+'_spec_hilbert.mat')
	A2 = d_v2['amp']
	P2 = d_v2['phase']
	S2 = d_v2['signal']

	d = dict()
	# A, P are freq x trials x time
	# signal is trials x time
	d['amp'] 	= np.concatenate(( A1[:,beta_frac_ind,:], A2[:,beta_pow_ind,:]), axis=1)
	d['phase'] 	= np.concatenate(( P1[:,beta_frac_ind,:], P2[:,beta_pow_ind,:]), axis=1)
	d['signal'] = np.concatenate(( S1[beta_frac_ind,:], S2[beta_pow_ind,:]), axis=0)
	sio.savemat(pre+cat_fname+'_sequRTs_spec_hilbert.mat',d)

def concatenate_kin_sig(bf_fname, bp_fname,Kin1, Kin2, bf_targ=[85,86,87], bp_targ=[84,85,86],lfp1=None, lfp2=None):
	pre = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'
	if lfp1 is None:
		bf_ind = sio.loadmat(pre+bf_fname)
		lfp_1 = bf_ind['LFP_label'][0,:]
	else:
		lfp_1 = lfp1

	beta_frac_ind = np.array([ i for i,t in enumerate(lfp_1) if t in bf_targ])

	if lfp2 is None:
		bp_ind = sio.loadmat(pre+bp_fname)
		lfp_2 = bp_ind['LFP_label'][0,:]
	else:
		lfp_2 = lfp2

	beta_pow_ind = np.array([ i for i,t in enumerate(lfp_2) if t in bp_targ])

	kin = np.vstack((Kin1[beta_frac_ind,:], Kin2[beta_pow_ind,:]))
	lfp = np.hstack((lfp_1[beta_frac_ind], lfp_2[beta_pow_ind] ))
	return kin, lfp

