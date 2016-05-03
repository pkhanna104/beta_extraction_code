from make_pytable_reach_bmi3d import Behav_Hand_Reach, manual_control_data, Kin_Traces
import tables
import numpy as np
import mc_metrics_go_nogo
from make_pytable_lfpmod_bmi3d import get_task_entries
from db import dbfunctions as dbfn


class Behav_Go_NoGo(tables.IsDescription):
	trial_type = tables.StringCol(2) 
	trial_outcome = tables.StringCol(14) 
	target_loc = tables.Float64Col(shape=(2,)) 
	reach_time = tables.Float64Col() 
	rxn_time = tables.Float64Col() 
	go_time = tables.Float64Col() 
	hold_time = tables.Float64Col() 
	reach_err = tables.Float64Col() 
	rew_rate = tables.Float64Col() 
	start_time = tables.Float64Col() 
	task_entry = tables.IntCol()
	trial_type = tables.StringCol(256) #
	trial_outcome = tables.StringCol(256) #
	nogo_time = tables.Float64Col() #

class gonogo_data(manual_control_data):

	def get_behavior(self):
			TE = self.task_entry_dict
			TE_start_inds = dict()
			TE_go_inds = dict()

			for i, tsk in enumerate(self.tasks):
				h5file = tables.openFile(self.tdy_str + self.behav_fname + tsk + '.h5', mode="w", title='Cart, behavior')
				table = h5file.createTable("/", 'behav', Behav_Go_NoGo, "Behavior Table")
				kin_table = h5file.createTable("/",'kin',Kin_Traces,"Kin Data")

				te_array = np.squeeze(TE[tsk])

				#Get for ALL trial types reach time, rxn time, hold time, reach error, rew rate, trial type distributions: 
				for j, te in enumerate(te_array):
					task_entry = dbfn.TaskEntry(te)
					nm = task_entry.name
					
					if self.system=='pk_mbp':
						hdf = tables.openFile('/Volumes/carmena/bmi3d/rawdata/hdf/'+nm+'.hdf')				
					elif self.system == 'sdh':
						hdf = tables.openFile('/storage/bmi3d/rawdata/hdf/'+nm+'.hdf')				
					else:
						print 'unrecognized system'

					#Trial Starts: 
					msg = hdf.root.task_msgs[:]['msg']
					msg_time = hdf.root.task_msgs[:]['time']
					start_ind = np.array([j for j, i in enumerate(msg) if i=='wait'])
					start_times = msg_time[start_ind]
					rew_times = np.array([t[1] for i, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

					TE_start_inds[tsk, te] = start_times
					TE_go_inds[tsk, te] = []
					#Categorize trials: 
					#Skip first trial: 
					for trial_ind, ix in enumerate(start_ind):
										#Get Trial outcome
						if ix+7 >= msg.shape[0]: #Very end of block
							print 'before: ', TE_start_inds[tsk, te].shape
							new_start_times = np.delete(TE_start_inds[tsk, te],[trial_ind])

							TE_start_inds[tsk, te] = new_start_times
							print 'must delete index! '
							print 'after: ', TE_start_inds[tsk, te].shape

						else:
							#Get Row of table: 
							trial = table.row
							#trial = dict()

							trial['task_entry'] = te
							trial['trial_type'] = tsk
							trial['start_time'] = start_times[trial_ind]

							kinrow = kin_table.row
							#kinrow = dict()
							kinrow['task_entry'] = te
							kinrow['start_time'] = start_times[trial_ind]

							if ix < len(msg)+7:
								if tsk in ['S1','S4','M1','M4']:
									trial, kinrow, TE_go_inds = mc_metrics.get_CO_metrics([trial_ind, ix], msg, msg_time, 
										trial, kinrow, TE_go_inds,rew_times, hdf,tsk, te)
								
								elif tsk in ['manualcontrol_memory']:
									trial, kinrow, TE_go_inds = mc_metrics.get_mc_mem_metrics([trial_ind, ix], msg, msg_time, 
										trial, kinrow, TE_go_inds, rew_times, hdf,tsk, te)

								elif tsk in ['manualcontrol_go_nogo_plus_gocatch']:
									trial, kinrow, TE_go_inds = mc_metrics_go_nogo.get_metrics([trial_ind, ix], msg, msg_time, 
										trial, kinrow, TE_go_inds, rew_times, hdf,tsk, te)

							trial.append()
							kinrow.append()
					kin_table.flush()		
					table.flush()
				h5file.close()
			sio.savemat(self.task_entry_dict_time_inds_fname, TE_start_inds)
			sio.savemat(self.task_entry_dict_go_times_fname, TE_go_inds)
			
			self.task_entry_dict_time_inds = TE_start_inds
			self.go_times = TE_go_inds

if __name__ == "__main__":

	kw = dict(te_min=6904)
	get_task_entries(tasks=['manualcontrol_go_nogo_plus_gocatch'],
		fname='task_entries_gonogo_start_march15.mat',**kw)

	d = dict(behav_file_name='cart_gonogo_beh',\
			neural_file_name = 'cart_gonogo_neu',\
			task_entry_dict_fname='task_entries_gonogo_start_march15.mat',\
			task_entry_dict_go = 'task_entries_gonogo_GO_march15.mat',\
			t_range=[2.25, 1.25],\
			tasks = ['manualcontrol_go_nogo_plus_gocatch']
			)

	mcd = gonogo_data(**d)

	#Get behav: 
	mcd.get_behavior()

	kw = dict(t_range=[2.25, 1.25], use_go_file=True, small_f_steps=True)

	jobs = []
	for tsk in mcd.tasks:
	#for tsk in ['S4']:
		print 'tsk: ', tsk
		p = multiprocessing.Process(target=mcd.make_neural,args=([tsk],),kwargs=kw)
		jobs.append(p)
		p.start()
		#mcd.make_neural(tsk)
