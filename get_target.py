import basic_spec
import classify
import parse
import numpy as np

# Get Targets: 
co_dates=['031814','031914','032514','032714','032814','032914','033014','033114','040114','042614','042714']
co_blocks = [['a'],['a'],['e'],['a'],['a'],['a'],['a'],['cdi'],['a'],['a'],['a']]
e_co = parse.epoch(align_code=5,dates=co_dates, blocks=co_blocks)
e_co.mat_file_check()
e_co.parse_all([67],parse_type='reward')
targ_Tco, e_co = basic_spec.cat_data(e_co, co_dates, co_blocks,[67])
y_co = classify.get_targ(targ_Tco)

#WM targets:
good_wm_dates=['033014','033114','040114']
good_wm_blocks = [['e'],['efgh'],['bc']]

e = parse.epoch(align_code=5,dates=good_wm_dates, blocks=good_wm_blocks)
e.mat_file_check()
e.parse_all([67],parse_type='reward')
targ_T, e = basic_spec.cat_data(e, good_wm_dates, good_wm_blocks,[67])
y_wm = classify.get_targ(targ_T)

#Save:
targets=dict()
targets['co']=y_co
targets['wm']=y_wm
sio.savemat('target_ind.mat',targets)