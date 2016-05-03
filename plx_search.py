import plxread
import scipy.io as sio
import os

def make_mat(plx_fname, mat_file, channels=range(65,193)):
	#plx_fname = plx_folder + anim_mth_day_yr + block + '.plx'
	print plx_fname, channels

	#Make sure file exists: 
	if os.path.exists(plx_fname):
		data = plxread.import_file(plx_fname,AD_channels=channels)
		sio.savemat(mat_file, data)
	else:
		print 'No Plexon file there!'
		print stop
		