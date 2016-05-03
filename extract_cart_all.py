if __name__ == '__main__':

    import sys
    arg_ind = int(sys.argv[1])

    if arg_ind == 0:
        d = dict(behav_file_name='pap_rev_cart_behav_upside_down',\
                       neural_file_name = 'pap_rev_cart_behav_upside_down_welch_neural',\
                       task_entry_dict_fname='task_entries_upside_down_cart.mat',\
                       task_entry_dict_go = 'task_entries_upside_down_cart.mat._start_inds.mat',\
                       t_range=[1, 2.5],\
                       spec_method='Welch',\
                       system='nucleus',\
                       tasks=['lfp_mod_mc_reach_out']
                       )

       import make_pytable_lfpmod_bmi3d as mp
       lfpd = mp.lfp_task_data(**d)
       lfpd.get_behavior(system='nucleus')
       lfpd.get_lfpmod_behavior(system='nucleus')
       lfpd.moving_window = [.251, .011]
       kw = dict(t_range=[2.5, 1], channels=[124])
       lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)
    
    ###########################
    # Cart Prep 1: 
    # te = [6281, 6282, 6283, 6284, 6285, 6286, 6287, 6288, 6294, 6295, 6298,
    #   6299, 6303, 6304, 6306, 6307, 6308], ntrials = [2328]
    elif arg_ind == 1:
        d = dict(behav_file_name='pap_rev_cart_behav_targ1',\
                       neural_file_name = 'pap_rev_cart_behav_targ1_welch_neural',\
                       task_entry_dict_fname='task_entries_targ1_cart.mat',\
                       task_entry_dict_go = 'task_entries_targ1_cart.mat._start_inds.mat',\
                       t_range=[1, 2.5],\
                       spec_method='Welch',\
                       system='nucleus',\
                       tasks=['lfp_mod_mc_reach_out']
                       )
       import make_pytable_lfpmod_bmi3d as mp
       lfpd = mp.lfp_task_data(**d)
       lfpd.get_behavior(system='nucleus')
       lfpd.get_lfpmod_behavior(system='nucleus')
       lfpd.moving_window = [.251, .011]
       kw = dict(t_range=[2.5, 1], channels=[124])
       lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)

    #For longer files: 
       lfpd.neural_file_name = 'pap_rev_cart_behav_targ1_welch_neural_long_times'
       lfpd.moving_window = [1.001, .011]
       kw = dict(t_range=[3.5, 2], channels=[124])
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
        d = dict(behav_file_name='pap_rev_cart_behav_targ1',\
                       neural_file_name = 'pap_rev_cart_behav_targ3_welch_neural',\
                       task_entry_dict_fname='task_entries_targ3_cart.mat',\
                       task_entry_dict_go = 'task_entries_targ3_cart.mat._start_inds.mat',\
                       t_range=[1, 2.5],\
                       spec_method='Welch',\
                       system='nucleus',\
                       tasks=['lfp_mod_mc_reach_out']
                       )
        import make_pytable_lfpmod_bmi3d as mp
        lfpd = mp.lfp_task_data(**d)
        lfpd.get_behavior(system='nucleus')
        lfpd.get_lfpmod_behavior(system='nucleus')
        lfpd.moving_window = [.251, .011]
        kw = dict(t_range=[2.5, 1], channels=[124])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)

        lfpd.neural_file_name = 'pap_rev_cart_behav_targ3_welch_neural_long_times'
        lfpd.moving_window = [1.001, .011]
        kw = dict(t_range=[3.5, 2], , channels=[124])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)


    ###########################
    # #Cart prep 7:
    # d['kin', 'c_t7'] = '2015-05-13kint7_beta_lfp_cart_behavlfp_mod_mc_reach_out.h5'
    # d['seq', 'c_t7'] = '2015-05-13t7_beta_lfp_cart_behavlfp_mod_mc_reach_out.h5'
    # te = [6964, 6965, 6966, 6967, 6968, 6969, 6970, 6971]
    # ntrials = 2685
    elif arg_ind == 7:
        d = dict(behav_file_name='pap_rev_cart_behav_targ7',\
                       neural_file_name = 'pap_rev_cart_behav_targ7_welch_neural',\
                       task_entry_dict_fname='task_entries_targ7_cart.mat',\
                       task_entry_dict_go = 'task_entries_targ7_cart.mat._start_inds.mat',\
                       t_range=[1, 2.5],\
                       spec_method='Welch',\
                       system='nucleus',\
                       tasks=['lfp_mod_mc_reach_out']
                       )
        import make_pytable_lfpmod_bmi3d as mp
        lfpd = mp.lfp_task_data(**d)
        lfpd.get_behavior(system='nucleus')
        lfpd.get_lfpmod_behavior(system='nucleus')
        lfpd.moving_window = [.251, .011]
        kw = dict(t_range=[2.5, 1], channels=[124])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)

        lfpd.neural_file_name = 'pap_rev_cart_behav_targ7_welch_neural_long_times'
        lfpd.moving_window = [1.001, .011]
        kw = dict(t_range=[3.5, 2], channels=[124])
        lfpd.make_neural(['lfp_mod_mc_reach_out'], **kw)