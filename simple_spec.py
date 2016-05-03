 
from tables import *
import numpy as np
import matplotlib.pyplot as plt
import nitime.algorithms as tsa
import scipy
import scipy.signal as scisig
import math
import datetime


def spike_rates(signal, t_ax, f_ax, tot_trials, fname):
	spike_rates = dict()
	unit_list = dict()
	n_d = len(np.unique([key[1] for i, key in enumerate(signal.keys())]))

	for d in range(n_d):
		keys = [ key for k, key in enumerate(signal.keys()) if key[1]==d ]
		n_b = len(np.unique([key[2] for i, key in enumerate(keys)]))

		units = np.unique([k[0] for i, k in enumerate(keys)])
		spk_rts = np.zeros(( len(units), tot_trials, len(t_ax) ))
		spk_rts[:] = np.NaN

		for b in range(n_b):
			#Get units for that block
			block_keys = [ key for k, key in enumerate(keys) if key[2]==b ]
			block_units = [(i, k[0]) for i, k in enumerate(block_keys)]
			
			for i_u, un in enumerate(block_keys):
				# Get unit index:
				ix_unit = np.nonzero(un[0]==units)[0]
				spk_rts[ix_unit, un[3], :] = get_rate(signal[un], t_ax)

		spike_rates[d] = spk_rts
		unit_list[d] = units
	return spike_rates, units


def get_rate(spike_times, t_ax, spike_bin = 0.020):

	#Correction for all positive t_ax: 
	if np.all(t_ax>0):
		t_ax = t_ax - 1.5 #assume 1.5 is the offset. 

	spike_rate = np.zeros((len(t_ax), ))
	for t, tm in enumerate(t_ax):
		ix = np.nonzero(np.logical_and(spike_times>=tm-(0.5*spike_bin), spike_times<tm+(0.5*spike_bin)))[0]
		spike_rate[t] = len(ix)/spike_bin
	return spike_rate

def Welch_specgram(data, movingwin=[0.2, 0.050], bp_filt = [10, 55], **kwargs):
	'''modeled from pwelch.m with addition of BP filtering b/w 10 Hz and 55 Hz beforehand

	data: 			format: time x trials (i.e. data[:,0] is 1 trial)

	movingwin:		 is in the format of [window, window_step] e.g. [0.2, 0.05] sec
	
	bp_filt: 		 inital BPfilter cutoffs

	kwargs: 
		pad: 		(padding factor for the FFT) - optional (can take values -1,0,1,2...). 
                     -1 corresponds to no padding, 0 corresponds to padding
                     to the next highest power of 2 etc.
 			      	 e.g. For N = 500, if PAD = -1, we do not pad; if PAD = 0, we pad the FFT
 			      	 to 512 points, if pad=1, we pad to 1024 points etc.
 			      	 Defaults to 0.
 		Fs:			(sampling frequency) 
 		fpass:		(frequency band to be used in the calculation in the form fmin fmax])- optional. 
                    Default all frequencies between 0 and Fs/2


	 Output:
        S       (spectrum in form time x frequency x channels/trials if trialave=0; 
                in the form time x frequency if trialave=1)
        t       (times)
        f       (frequencies)

	'''

	if 'Fs' in kwargs.keys():
		Fs = kwargs['Fs']
	else:
		Fs = 1000

	if 'fpass' in kwargs.keys():
		fpass = kwargs['fpass']
	else:
		fpass = [0, 100]

	if 'pad' in kwargs.keys():
		pad = kwargs['pad']
	else:
		pad = 0

	#First BP filter: 
	nyq = 0.5* Fs
	bw_b, bw_a = scisig.butter(5, [bp_filt[0]/nyq, bp_filt[1]/nyq], btype='band')

	#Use default padding:
	data_filt = scisig.filtfilt(bw_b, bw_a, data, axis=0)

	num_trials = data_filt.shape[1]
	N = data_filt.shape[0] #ms of trials
	Nwin=round(Fs*movingwin[0])
	Nstep=round(Fs*movingwin[1])

	#Choose the power of 2 > Nwin (or add pad if needed)
	t_power = 1
	while 2**t_power < Nwin:
		t_power +=1
	nfft = 2**(t_power+pad)

	winstart=np.arange(0,N-Nwin,Nstep)
	nw=len(winstart)
	
	#Dimensions of S: trials x num_win (t) x f

	for n in range(nw):
		w_start = int(winstart[int(n)])

		#Window x Samples: 
		datawin=data_filt[w_start:w_start+Nwin,:].T

		if n==0:
			f_test, _ = scisig.welch(np.zeros((Nwin, num_trials)), fs=Fs, nperseg=Nwin, nfft=nfft, axis=0)
			S = np.zeros(( num_trials, nw,  len(f_test)))

		f , psd_est = scisig.welch(datawin, fs=Fs, nperseg=Nwin, nfft=nfft, axis=1)
		S[:, n, :] = psd_est

	t=(winstart+round(Nwin/2))/float(Fs)

	return S, f, t


	
def MTM_specgram(data,movingwin=[0.2, 0.050],**kwargs):
	'''modeled from mtspecgramc.m 

	data: 			format: time x trials (i.e. data[:,0] is 1 trial)
	movingwin:		 is in the format of [window, window_step] e.g. [0.2, 0.05] sec
	kwargs: 
		tapers:		in the form of [TW, K] e.g. [5/2 5]
		pad: 		(padding factor for the FFT) - optional (can take values -1,0,1,2...). 
                     -1 corresponds to no padding, 0 corresponds to padding
                     to the next highest power of 2 etc.
 			      	 e.g. For N = 500, if PAD = -1, we do not pad; if PAD = 0, we pad the FFT
 			      	 to 512 points, if pad=1, we pad to 1024 points etc.
 			      	 Defaults to 0.
 		Fs:			(sampling frequency) 
 		fpass:		(frequency band to be used in the calculation in the form fmin fmax])- optional. 
                    Default all frequencies between 0 and Fs/2
		trialave    (average over trials/channels when 1


	 Output:
        S       (spectrum in form time x frequency x channels/trials if trialave=0; 
                in the form time x frequency if trialave=1)
        t       (times)
        f       (frequencies)

	'''

	if 'tapers' in kwargs.keys():
		tapers = kwargs['tapers']
	else:
		tapers = [5/2, 5]

	if 'pad' in kwargs.keys():
		pad = kwargs['pad']
	else:
		pad = 0

	if 'Fs' in kwargs.keys():
		Fs = kwargs['Fs']
	else:
		Fs = 1000

	if 'fpass' in kwargs.keys():
		fpass = kwargs['fpass']
	else:
		fpass = [0, 100]

	if 'trialave' in kwargs.keys():
		trialave = kwargs['trialave']
	else:
		trialave = 0

	num_trials = data.shape[1]

	N = data.shape[0] #ms of trials

	Nwin=round(Fs*movingwin[0])
	Nstep=round(Fs*movingwin[1])

	t_power = 1
	while 2**t_power < Nwin:
		t_power +=1
	nfft = 2**(t_power+pad)

	#f=getfgrid(Fs,nfft,fpass)wi
	#f = np.arange(0,501,5)
	winstart=np.arange(0,N-Nwin,Nstep)
	nw=len(winstart)
	
	#Dimensions of S: trials x num_win (t) x f

	for n in range(nw):

		datawin=data[int(winstart[int(n)]):int(winstart[int(n)])+int(Nwin),:].T
		if  datawin.shape[1] < nfft :
			dat = np.zeros(( datawin.shape[0], nfft ))
			pad = (nfft - datawin.shape[1])
			if pad%2: #Odd:
				pad1 = pad2 = np.floor(pad/2.)
			
			elif not pad%2:
				pad1 = pad2 = pad/2
			
			dat[:, pad1:(datawin.shape[1]+pad2)] = datawin
		
		elif nfft == datawin.shape[1]:
			dat = datawin
		
		else:
			raise Exception("Not implemented yet...")

		if 'small_f_steps' in kwargs.keys():
			f, psd_est, nu = tsa.multi_taper_psd(dat,Fs=Fs, NFFT=nfft)
			
		else:
			f, psd_est, nu = tsa.multi_taper_psd(dat,Fs=Fs)

		if n==0:
			S = np.zeros(( num_trials, nw, len(f[f<fpass[1]]) ))

		#print len(f), psd_est.shape, len(nu), S.shape, len(f<fpass[1])
		S[:,n,:] = psd_est[:,f<fpass[1]]

	t=(winstart+round(Nwin/2))/float(Fs)

	if trialave:
		S = np.mean(S,axis=0)

	return S, f, t


def DFT_PSD(data,movingwin=[0.201, 0.051], Fs = 1000, pad=0, fpass=[1,100]):
	'''Discrete Fourier Transform
		Input: 
			data: format is np.array that is  time_window x samples
		
		'''
	num_trials = data.shape[1]
	N = data.shape[0] #ms of trials
	Nwin=round(Fs*movingwin[0])
	Nstep=round(Fs*movingwin[1])
	
	winstart=np.arange(0,N-Nwin,Nstep)
	nw=len(winstart)

	f = np.fft.fftfreq(int(movingwin[0]*Fs))
	f = Fs*f[f>=0]
	f_ind = scipy.logical_and(f>=fpass[0], f<=fpass[1])

	#set(f[f>=fpass[0]] ) & set(f[f<=fpass[1]])
	#f_ind = np.array(list(f_ind))

	S = np.zeros(( num_trials, nw, sum(f_ind) ))

	for n in range(nw):
		datawin=data[winstart[n]:winstart[n]+Nwin,:]
		sp = np.fft.rfft(datawin.T)
		psd_est = abs(sp)**2
		S[:,n,:] = psd_est[:,f_ind] 

	t=(winstart+round(Nwin/2))/float(Fs)
	return S, f[f_ind], t

def hilbert_transform(data,bandpass=[25,40],Fs=1000):
	'''data input as time x samples'''
	#bandpass=[25,40]
	#Fs=1000
	upper = bandpass[1]/float(Fs/2)
	lower = bandpass[0]/float(Fs/2)
	order,wn = scisig.buttord([lower, upper],[np.max([0, lower-0.01]), np.min([upper+0.01, 1])],3,20)
	b, a = scisig.butter(np.min([order,9]), wn, btype = 'bandpass')
	
	amp = np.zeros((data.shape[0],data.shape[1]))
	phase = np.zeros((data.shape[0],data.shape[1]))
	for i, d in enumerate(data.T):
		y = scisig.filtfilt(b,a,d)
		z = scisig.hilbert(y)
		amp[:,i] = abs(z)
		phase[:,i] = phz(z)
	return amp, phase

def phz(x):
	phase = np.zeros(( len(x), ))
	for i,xi in enumerate(x):
		phase[i] = math.atan2(np.real(xi),np.imag(xi))
	return phase

def getfgrid(Fs,nfft,fpass):
	df=Fs/(float(nfft))
	f=np.arange(0,Fs,df)
	f = f[0:nfft]
	t1 = f>=fpass[0]
	t2 = f<=fpass[1]
	findx = np.nonzero(np.logical_and(t1,t2))[0]
	f=f[findx]
	return f