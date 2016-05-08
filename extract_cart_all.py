
if __name__ == '__main__':

    import sys
    arg_ind = int(sys.argv[1])


    d = dict(neural_sig_only=True, t_range=[1, 2.5], system='nucleus', tasks=['lfp_mod_mc_reach_out'],
        spec_method='Welch')

    if arg_ind == 0:
        d2 = dict(behav_file_name='pap_rev_cart_behav_upside_down',\
                       neural_file_name = 'pap_rev_cart_behav_upside_down_welch_neural',\
                       task_entry_dict_fname='task_entries_upside_down_cart.mat',\
                       task_entry_dict_go = 'task_entries_upside_down_cart.mat._start_inds.mat')

        d3 = d.update(d2)

        import make_pytable_lfpmod_bmi3d as mp
        lfpd = mp.lfp_task_data(**d3)
        lfpd.get_behavior(system='nucleus')
        lfpd.get_lfpmod_behavior(system='nucleus')
        lfpd.moving_window = [.251, .011]
        kw = dict(t_range=[2.5, 1], channels=[124, 33, 1])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)
    
    ###########################
    # Cart Prep 1: 
    # te = [6281, 6282, 6283, 6284, 6285, 6286, 6287, 6288, 6294, 6295, 6298,
    #   6299, 6303, 6304, 6306, 6307, 6308], ntrials = [2328]
    elif arg_ind == 1:
        d2 = dict(behav_file_name='pap_rev_cart_behav_targ1',\
                       neural_file_name = 'pap_rev_cart_behav_targ1_welch_neural',\
                       task_entry_dict_fname='task_entries_targ1_cart.mat',\
                       task_entry_dict_go = 'task_entries_targ1_cart.mat._start_inds.mat',\
                       )
        d3 = d.update(d2)

        import make_pytable_lfpmod_bmi3d as mp
        lfpd = mp.lfp_task_data(**d3)
        lfpd.get_behavior(system='nucleus')
        lfpd.get_lfpmod_behavior(system='nucleus')
        lfpd.moving_window = [.251, .011]

        kw = dict(t_range=[2.5, 1], channels=[124, 33, 1])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)

        #For longer files: 
        lfpd.neur_fname = 'pap_rev_cart_behav_targ1_welch_neural_long_times'
        lfpd.moving_window = [1.001, .011]

        kw = dict(t_range=[3.5, 2], channels=[124, 33, 1])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)

   # Old files: 
    # #Cart prep 1:
    # d['kin', 'c_t1'] = '2015-01-26t1_lfp_cart_behavlfp_mod_mc_reach_out.h5'
    # d['seq', 'c_t1'] = '2015-01-26kint1_lfp_cart_behavlfp_mod_mc_reach_out.h5'


    ###########################
    #Cart prep 3: 
    # d['kin', 'c_t3'] = '2015-01-27kint3_lfp_cart_behavlfp_mod_mc_reach_out.h5'
    # d['seq', 'c_t3'] = '2015-01-27t3_lfp_cart_behavlfp_mod_mc_reach_out.h5'
    # te = [6314, 6315, 6317, 6318, 6353, 6354]
    # ntrials = 2036
    elif arg_ind == 3:
        d2 = dict(behav_file_name='pap_rev_cart_behav_targ3',\
                       neural_file_name = 'pap_rev_cart_behav_targ3_welch_neural',\
                       task_entry_dict_fname='task_entries_targ3_cart.mat',\
                       task_entry_dict_go = 'task_entries_targ3_cart.mat._start_inds.mat',\
                       )
        d3 = d.update(d2)

        import make_pytable_lfpmod_bmi3d as mp
        lfpd = mp.lfp_task_data(**d3)
        lfpd.get_behavior(system='nucleus')
        lfpd.get_lfpmod_behavior(system='nucleus')
        lfpd.moving_window = [.251, .011]
        kw = dict(t_range=[2.5, 1], channels=[124, 33, 1])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)

        lfpd.neur_fname = 'pap_rev_cart_behav_targ3_welch_neural_long_times'
        lfpd.moving_window = [1.001, .011]
        kw = dict(t_range=[3.5, 2], channels=[124, 33, 1])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)


    ###########################
    # #Cart prep 7:
    # d['kin', 'c_t7'] = '2015-05-13kint7_beta_lfp_cart_behavlfp_mod_mc_reach_out.h5'
    # d['seq', 'c_t7'] = '2015-05-13t7_beta_lfp_cart_behavlfp_mod_mc_reach_out.h5'
    # te = [6964, 6965, 6966, 6967, 6968, 6969, 6970, 6971]
    # ntrials = 2685
    elif arg_ind == 7:
        d2 = dict(behav_file_name='pap_rev_cart_behav_targ7',\
                       neural_file_name = 'pap_rev_cart_behav_targ7_welch_neural',\
                       task_entry_dict_fname='task_entries_targ7_cart.mat',\
                       task_entry_dict_go = 'task_entries_targ7_cart.mat._start_inds.mat',\
                       )
        d3 = d.update(d2)

        import make_pytable_lfpmod_bmi3d as mp
        lfpd = mp.lfp_task_data(**d3)
        lfpd.get_behavior(system='nucleus')
        lfpd.get_lfpmod_behavior(system='nucleus')
        lfpd.moving_window = [.251, .011]
        kw = dict(t_range=[2.5, 1], channels=[124, 33, 1])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)

        lfpd.neur_fname = 'pap_rev_cart_behav_targ7_welch_neural_long_times'
        lfpd.moving_window = [1.001, .011]
        kw = dict(t_range=[3.5, 2], channels=[124, 33, 1])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)

    elif arg_ind == 10:
        d = dict(behav_file_name='pap_rev_new_MC_cart_behav',\
               neural_file_name = 'pap_rev_new_MC_cart_welch_neural',\
               task_entry_dict_fname='task_entries_manual_control_cart.mat',\
               task_entry_dict_go = 'task_entries_manual_control_cart._start_inds.mat',\
               t_range=[1, 2.5],\
               spec_method='Welch',\
               system='nucleus', \
               tasks=['S1', 'M1'],
               neural_sig_only=True
               )
        import make_pytable_reach_bmi3d as mp
        mcd = mp.manual_control_data(**d)
        mcd.get_behavior()
        kw = dict(t_range=[1,2.5],use_go_file=True, spec_method='Welch')
        mcd.moving_window = [.251, .011]
        jobs = []
        import multiprocessing
        for tsk in mcd.tasks:
            print 'tsk: ', tsk
            p = multiprocessing.Process(target=mcd.make_neural,args=([tsk],),kwargs=kw)
            jobs.append(p)
            p.start()

    elif arg_ind == 11:
        d2 = dict(behav_file_name='pap_rev_cart_xy_behav',\
               neural_file_name = 'pap_rev_cart_xy_welch_neural',\
               task_entry_dict_fname='task_entries_cart_xy.mat',\
               task_entry_dict_go = 'ttask_entries_cart_xy_GO.mat',\
               )
        d3 = d.update(d2)
        
        import make_pytable_reach_bmi3d as mp
        mcd = mp.manual_control_data(**d3)
        mcd.get_behavior()
        kw = dict(t_range=[1,2.5],use_go_file=True, spec_method='Welch')
        mcd.moving_window = [.251, .011]
        mcd.make_neural([tsk],**kw)

        


