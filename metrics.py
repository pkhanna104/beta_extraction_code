

def trials_per_minute(e,window=10):
	for d,day in enumerate(e.file_feats['dates']):
		for b,block in enumerate(e.file_feats['blocks'][d][0]):
			fname = e.file_feats['mat_data_path'] + e.file_feats['anim'] + day + block + '.mat'
			data = sio.loadmat(fname)
			strobed = data['Strobed']
			adxx = np.zeros((data['AD65'].shape))
			rew = np.arary([strobed[c,0] for c,code in enumerate(strobed[:,1]) if code == 11])

			rew_ind = int(np.round(rew * e.file_feats['fs']))
			adxx[rew_ind] = 1

			win = window*e.file_feats['fs']
			adxx_avg = movingaverage(adxx,win)
			return adxx_avg*window/60 #convert to trials per minute

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
