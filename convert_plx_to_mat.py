import os
import plx_search
import scipy.io as sio

# dates = ['042914','050214','050314','050414']
# blocks=[['ij'],['abcdefghij'],['efgh'],['abcdefg']]


# dates = ['042914','043014','050214','042914','043014','050314','050414','050514'] 
# blocks=[['ij'],['ab'],['abcdefghij'],['abcdefgh'],['cdefghi'], ['efg'],['abdefg'],['abcdefghij']]

dates = ['050514','050614','050714']
blocks = [['abcdefghij'],['abdeghi'],['abcdefghi']]

for d,day in enumerate(dates):
	for b,block in enumerate(blocks[d][0]):

		filename = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/seba' + day + block + '.mat'
		
		if os.path.isfile(filename):
			try:
				d=sio.loadmat(filename,variable_names='Strobed')
			except:
				os.remove(filename)

		if not os.path.isfile(filename):
			print 'making .mat for ' + day + block
			try:
				plx_search.make_mat('/Volumes/carmena/' + 'seba/seba' + day + '/', block , '/Volumes/TimeMachineBackups/Seba_MC_mat_files/', 'seba' + day)
			except:
				plx_search.make_mat('/Volumes/carmena/' + 'seba/seba' + day + '/'+ 'map_data/', block, '/Volumes/TimeMachineBackups/Seba_MC_mat_files/','seba' + day)
		

			
