import os

def find_folder(root_dir,folder_name, match_full_name = True):
    '''Returns all the dirs of folders with folder_name only'''
    pass

def _extract_folder_names(root_dir, parent_folder_keyword = None, child_folder_keyword = None):
    '''Returns the string which is the folder name either after parent_folder_keyword
    or before child_folder_keyword '''
    pass

def _get_folders_in_folders(fldr_path):
    fldrs = [al for al in os.listdir(fldr_path) if os.path.isdir(os.path.join(fldr_path, al))]
    return fldrs

def _get_single_file_in_folder(folder_path):
    pass

def _check_folder_path_exixt(fldr_path):
    exist = os.path.isdir(fldr_path)
    if exist == False:
        raise Exception("The folder path {} does not exist.".format(fldr_path))
    

def get_experiment_folders(root_dir, experiment_group_name, include_bechmark = True):
    '''Returns a list of folders of the central plot event file experiments '''
    central_plot_keyword = "Central"
    central_experiments = []
    becnchmark_experiments = []

    check_folder_path_exixt(root_dir)

    all_ex_groups = get_folders_in_folders(root_dir)

    the_exp_folders = [fldr for fldr in all_ex_groups if experiment_group_name in fldr]

    assert len(the_exp_folders) == 2, \
        "There is not two folders for the exp. Folders found: {}".format(the_exp_folders)
    
    central_folder = [fldr for fldr in the_exp_folders if central_plot_keyword.lower() in fldr.lower()]

    assert len(central_folder) == 1, "Number of central folders not one. Folders: {}".format(central_folder)

    central_experiments = get_folders_in_folders(central_folder[0])

    if include_bechmark:
        non_central_exp_fldr = the_exp_folders.remove(central_folder)
        becnchmark_experiments = find_folder(non_central_exp_fldr[0], "benchmark")
    
    return becnchmark_experiments, central_experiments

def _get_benchmark_event_files(bench_folders):
    '''Get the event files inside the subfolders with the highest suffix number 
    of the bench folders '''
    pass

def get_event_files(root_folder, experiment_group_name, include_bechmark = True):
    benchmark_fldrs, central_exp_fldrs = get_experiment_folders(experiment_group_name, include_bechmark)

    len_cen = len(central_exp_fldrs)
    len_ben = len(benchmark_fldrs)
    print("Number of experiment folders: {}  | Number of benchmark folders: {}".\
        format(len_cen, len_ben))

    if include_bechmark:
        if len_cen != len_ben:
            print("WARNING: Number of benchmark experiments does not match number of cenral experiments")
    
    if len(benchmark_fldrs) !=0 and include_bechmark:
        bench_events = get_benchmark_event_files(bench_fldrs)
    else:
        bench_events = []
    
    cent_exp_events = [get_event_files(fldr) for fldr in central_exp_fldrs]

    return bench_events, cent_exp_events



if __name__ == "__main__":
    root_fldr = '/home/desktop123/Desktop/Gridworld2'
    exp_grp_name = 'Exp0CN'

    bench_events, central_events = get_event_files(root_fldr, exp_grp_name)

    print(central_events)


