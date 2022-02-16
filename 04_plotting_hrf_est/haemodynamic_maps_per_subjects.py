""" Plot the haemodynamic delay parameter boxplot per region. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import subprocess
import time
from glob import glob
import argparse
import pickle
import json
import pandas as pd


temgesic_group = ['S00634_1558', 'S00651_1695', 'S00669_1795', 'S00726_2083',
                  'S00748_2205', 'S00791_2518', 'S00805_2631']


# Main
if __name__ == '__main__':

    # python3 haemodynamic_maps_per_subjects.py --bids-root-dir /media/veracrypt1/synchropioid/fmri_nifti_dir/ --results-dir ../02_hrf_est/results_hrf_estimation/ --best-params-file decomp_params/best_single_subject_decomp_params.json --output-dir output_dir --task-filter only_hb_rest --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--bids-root-dir', type=str,
                        default='fmri_nifti_dir',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--results-dir', type=str, default='results_dir',
                        help='Set the name of the results directory.')
    parser.add_argument('--output-dir', type=str, default='output_dir',
                        help='Set the name of the output directory.')
    parser.add_argument('--best-params-file', type=str,
                        default='best_group_decomp_params.json',
                        help='Load the best decomposition parameters.')
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
    # Collect the functional data
    participants_fname = os.path.join(args.bids_root_dir, 'participants.tsv')
    participants_path = os.path.abspath(os.path.normpath(participants_fname))
    participants = pd.read_csv(participants_path, sep='\t')
    sub_tag_convert = dict(zip(participants['participant_id'],
                            participants['DICOM tag']))

    with open(args.best_params_file, 'r') as jsonfile:
        best_params = json.load(jsonfile)

    if args.verbose:
            print(f"Best parameters ({best_params}) loaded from "
                  f"'{args.best_params_file}'")

    results_dir = os.path.abspath(os.path.normpath(args.results_dir))
    decomp_dirs = set([os.path.dirname(pickle_fname) for pickle_fname in
                       glob(f"{results_dir}/**/*.pkl", recursive=True)])

    if len(decomp_dirs) == 0:
        raise ValueError(f"No results found under '{args.results_dir}'")

    if args.verbose:
        print(f"Saving vascular maps under '{args.output_dir}' directory")

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    ###########################################################################
    # Compute statistic
    for decomp_dir in decomp_dirs:

        results_path = os.path.join(decomp_dir, 'results.pkl')
        with open(results_path, 'rb') as pfile:
            results = pickle.load(pfile)

        metadata_path = os.path.join(decomp_dir, 'metadata.json')
        with open(metadata_path, 'r') as jsonfile:
            metadata = json.load(jsonfile)

        try:

            a_hat_img = results['a_hat_img']
            t_r = metadata['t_r']
            lbda = metadata['decomp_params']['lbda']
            n_atoms = metadata['decomp_params']['n_atoms']
            run = metadata['n_run']
            sub = (metadata['sub_tag'] if 'sub_tag' in metadata
                                       else metadata['sub_tags'])
            group = metadata['group_label']

            shared_maps = True if isinstance(a_hat_img, list) else False

            if lbda == best_params['lbda']:

                if t_r in valid_tr:

                    if shared_maps:
                        for sub_, a_hat_img_ in zip(sub, a_hat_img):
                            filename = (f"{sub_}_run-{run}_group-{group}_"
                                        f"tr-{t_r}_shared_maps_hrfmaps.nii.gz")
                            filepath = os.path.join(args.output_dir, filename)
                            a_hat_img_.to_filename(filepath)

                    else:
                        filename = (f"{sub}_run-{run}_group-{group}_tr-{t_r}_"
                                    f"hrfmaps.nii.gz")
                        filepath = os.path.join(args.output_dir, filename)
                        a_hat_img.to_filename(filepath)

        except KeyError as e:
            continue

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
