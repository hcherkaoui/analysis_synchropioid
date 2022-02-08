""" Plot the difference on vascular maps. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import time
import subprocess
import argparse
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nilearn import input_data, plotting


# Main
if __name__ == '__main__':

    # python3 plot_temgesics_vs_control_group.py --vascular-maps-dir output_dir  --bids-root-dir /media/veracrypt1/synchropioid/fmri_nifti_dir/ --plots-dir plots --task-filter only_hb_rest --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--vascular-maps-dir', type=str, default='output_dir',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--bids-root-dir', type=str,
                        default='fmri_nifti_dir',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Plots directory.')
    parser.add_argument('--task-filter', type=str,
                        default='all_task',
                        help='Filter the fMRI task loaded, valid options are '
                             '["only_hb_rest", "only_rest", "all_task"].')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if args.task_filter == 'only_hb_rest':
        valid_tr = [0.8]

    elif args.task_filter == 'only_rest':
        valid_tr = [2.0]

    elif args.task_filter == 'all_task':
        valid_tr = [0.8, 2.0]

    else:
        valid_tr = [0.8, 2.0]

    ###########################################################################
    # Collect the seed base analysis z-maps
    participants_fname = os.path.join(args.bids_root_dir, 'participants.tsv')
    participants_path = os.path.abspath(os.path.normpath(participants_fname))
    participants = pd.read_csv(participants_path, sep='\t')
    sub_tag_convert = dict(zip(participants['participant_id'],
                               participants['DICOM tag']))

    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir, exist_ok=True)

    template_vascular_maps_paths = os.path.join(args.vascular_maps_dir,
                                                f"sub-*_run-*_group-*_"
                                                f"tr-*_*.nii.gz")
    vascular_maps_paths = glob(template_vascular_maps_paths)

    masker = input_data.NiftiMasker()
    masker.fit(vascular_maps_paths)

    i = 0
    columns = ['sub', 'run', 'group', 'vascular_map']
    vascular_maps = pd.DataFrame(columns=columns)
    for vascular_maps_path in vascular_maps_paths:

        vascular_maps_path = os.path.normpath(vascular_maps_path)
        vascular_maps_name = os.path.basename(vascular_maps_path)
        chunks = vascular_maps_name.split('_')
        sub, run, group, t_r = chunks[0], chunks[1], chunks[2], chunks[3]

        sub = int(sub.split('-')[-1])
        run = f"Run-{int(run.split('-')[-1])}"
        group = group.split('-')[-1]
        t_r = float(t_r.split('-')[-1])

        vascular_map = masker.transform_single_imgs(vascular_maps_path)

        if t_r in valid_tr:

            vascular_maps.loc[i] = [sub, run, group, vascular_map.flatten()]

            if args.verbose:
                print(f"\r[{i+1:02d}/{len(vascular_maps_paths):02d}] Vascular "
                    f"maps extraction for '{vascular_maps_path}'", end='')

            i += 1

    runs = np.sort(np.unique(vascular_maps['run']))
    filter_temgesics = vascular_maps['group'] == 'temgesic'
    n_sub = len(np.unique(vascular_maps[filter_temgesics]['sub']))

    ###########################################################################
    # Plotting
    nrows, ncols = len(runs), n_sub
    fig, axis = plt.subplots(nrows, ncols, figsize=(ncols * 15, nrows * 2))

    # plot the vascular difference maps
    for row_idx_ax, run in zip(range(nrows), runs):

        filter = vascular_maps['run'] == run
        filter_control = filter & (vascular_maps['group'] == 'control')

        a_control = list(vascular_maps[filter_control]['vascular_map'])
        mean_controls = np.mean(np.r_[a_control], axis=0)

        filter_temgesics = filter & (vascular_maps['group'] == 'temgesic')
        temgesics_unique_subjects = \
                            np.unique(vascular_maps[filter_temgesics]['sub'])

        for col_idx_ax, unique_sub in zip(range(ncols),
                                          temgesics_unique_subjects):

            temgesics_subjects = vascular_maps[filter_temgesics]['sub']
            filter_subject = temgesics_subjects == unique_sub
            filter_temgesic = filter_temgesics & filter_subject

            temgesic = vascular_maps[filter_temgesic]['vascular_map']
            temgesic = np.array(list(temgesic)).flatten()

            stats_img = masker.inverse_transform(temgesic - mean_controls)

            sub_tag = sub_tag_convert[f"sub-{unique_sub}"]
            title = f"{sub_tag}|{run}"
            plotting.plot_stat_map(stats_img,
                                    title=title,
                                    cmap='bwr', display_mode='z',
                                    colorbar=True,
                                    cut_coords=np.linspace(-40, 70, 10),
                                    axes=axis[row_idx_ax, col_idx_ax],
                                    vmax=1.5)

    # deactivate axis lines
    for row_idx_ax in range(nrows):
        for col_idx_ax in range(ncols):
            axis[row_idx_ax, col_idx_ax].set_axis_off()

    filename = f"temgesic_diff_vascular_maps_per_run"
    filename += f"_from_{args.vascular_maps_dir}"

    filepath = os.path.join(args.plots_dir, filename + '.pdf')
    print(f"Saving plot at '{filepath}'")
    plt.savefig(filepath, dpi=300)
    subprocess.call(f"pdfcrop {filepath} {filepath}",
                    shell=True)

    filepath = os.path.join(args.plots_dir, filename + '.png')
    print(f"Saving plot at '{filepath}'")
    plt.savefig(filepath, dpi=300)


    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
