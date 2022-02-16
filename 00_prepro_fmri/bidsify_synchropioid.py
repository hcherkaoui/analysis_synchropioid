#!/usr/bin/env python3
""" Tool to format the Synchropioid DICOM data into a valid BIDS dataset.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import time
import shutil
import subprocess
import argparse
from glob import glob
from joblib import Parallel, delayed
import json
import numpy as np
import pandas as pd


t0 = time.time()


DESCR_ = {
    "Name": "Synchropioid",
    "BIDSVersion": "1.0.2",
    "License": "",
    "Authors": [
        "LEROY Claire",
        "TOURNIER Nicolas",
        "BOTTLAENDER Michel"
    ],
    "Acknowledgements": "BARON Christine, BEAU Stephanie, BRULON Vincent, CAILLE Fabien, CHERKAOUI Hamza, COMTAT Claude, EYL Berangere, FERNANDEZ Brice, FERRAND Charlotte, GERVAIS Philippe, GOISLARD Maud, GOUTAL Sebastien, MAISON Veronique, MANCIOT Christine, PETIT Marion",
    "HowToAcknowledge": "",
    "Funding": [
        "",
        "",
        ""
    ],
    "ReferencesAndLinks": [
        "",
        "",
        ""
    ],
    "DatasetDOI": ""
    }


REST_DESCR_ = {
    "RepetitionTime": 2.0,
    "TaskName": "Resting State",
    "TaskDescription": "Resting State fMRI with TR=2.0s"
    }


HBREST_DESCR_ = {
    "RepetitionTime": 0.80,
    "TaskName": "'Hyper-Band' Resting State",
    "TaskDescription": "Resting State fMRI with TR=0.8s"
    }


CHANGES_ = """
0.0.1 2020-11-30
 - First upload
0.0.2 2021-08-20
 - Adding TR=2s EPI sequences (resting state)
"""


README_ = """
Synchropioid dataset

Objectif

The objectif of the project is to exhibit the temporal and spatial
synchronicity between the transition of the blood brain barrier of the
buprenorphine, its receptor occupancy and the haemodynamic and pharmacologic
measured response in the humain.
"""


DICOM_PATTERNS = {
    'rest_1': '0000*_EPI-Rest-*-n1',
    'rest_2': '0000*_EPI-Rest-*-n2',
    'rest_3': '0000*_EPI-Rest-*-n3',
    'rest_4': '0000*_EPI-Rest-*-n4',
    'rest_5': '0000*_EPI-Rest-*-n5',
    'hbrest_3': '0000*_HB-EPI-Rest-*-n3',
    'hbrest_5': '0000*_HB-EPI-Rest-*-n5',
    't1': '0000*Sag-3D-T1w*',
    'topup': '0000*EPI-RevBlips*' ,
    'b0_magnitude': '0000*B0Map-EPI',
}


def join_dirs(l_paths):
    """ Joined all paths together.
    """
    if len(l_paths) == 1:
        return l_paths
    joined_path = l_paths[0]
    for path in l_paths[1:]:
        joined_path = os.path.join(joined_path, path)

    return joined_path


def get_sub_dicom_filenames(root_dicom_dir):
    """ Get the files tree of the found dicom files udner the given dir.
    """
    dicom_tree = {}
    mr_dir = root_dicom_dir

    for name, pattern in DICOM_PATTERNS.items():
        l_dir = glob(os.path.join(mr_dir, pattern))
        dicom_tree[name] = l_dir[0] if l_dir else None

    return dicom_tree


def get_sub_bids_filenames(root_bids_dir, idx_sub):
    """ Get the valid BIDS filename structure given the root_bids_dir and the
    index of the subject.
    """
    sub_bids_tree = {  # define the correspondance logic between DICOM and BIDS
        'rest_1': join_dirs([root_bids_dir, f'sub-{idx_sub:02d}',
                           'func', f'sub-{idx_sub:02d}_task-rest_run-1_bold']),
        'rest_2': join_dirs([root_bids_dir, f'sub-{idx_sub:02d}',
                           'func', f'sub-{idx_sub:02d}_task-rest_run-2_bold']),
        'rest_3': join_dirs([root_bids_dir, f'sub-{idx_sub:02d}',
                           'func', f'sub-{idx_sub:02d}_task-rest_run-3_bold']),
        'rest_4': join_dirs([root_bids_dir, f'sub-{idx_sub:02d}',
                           'func', f'sub-{idx_sub:02d}_task-rest_run-4_bold']),
        'rest_5': join_dirs([root_bids_dir, f'sub-{idx_sub:02d}',
                           'func', f'sub-{idx_sub:02d}_task-rest_run-5_bold']),
        'hbrest_3': join_dirs([root_bids_dir, f'sub-{idx_sub:02d}',
                        'func', f'sub-{idx_sub:02d}_task-hbrest_run-3_bold']),
        'hbrest_5': join_dirs([root_bids_dir, f'sub-{idx_sub:02d}',
                        'func', f'sub-{idx_sub:02d}_task-hbrest_run-5_bold']),
        't1': join_dirs([root_bids_dir, f'sub-{idx_sub:02d}', 'anat',
                                                    f'sub-{idx_sub:02d}_T1w']),
        'topup': join_dirs([root_bids_dir, f'sub-{idx_sub:02d}',
                                      'fmap', f'sub-{idx_sub:02d}_dir-0_epi']),
        'b0_magnitude': join_dirs([root_bids_dir, f'sub-{idx_sub:02d}',
                                      'fmap', f'sub-{idx_sub:02d}_magnitude']),
    }
    return sub_bids_tree


def transform_dicomfiles_to_bidsfiles(sub_dicom_tree, sub_bids_tree,
                                      force_conversion=False, verbose=False):
    """ Produce the convertion of the DICOM to the Nifti and re-arrange the
    files as specify by the given BIDS-tree.
    """
    for name, dicom_dir in sub_dicom_tree.items():

        if dicom_dir is not None:

            # prepare directories
            dir_ouput = os.path.dirname(sub_bids_tree[name])
            if not os.path.exists(dir_ouput):
                os.makedirs(dir_ouput)

            filename_ouput = os.path.basename(sub_bids_tree[name])
            dir_input = dicom_dir
            bids_filepath = sub_bids_tree[name] + '.nii.gz'

            sub_tag = os.path.basename(bids_filepath).split('_')[0]

            cmd = (f'./dcm2niix -b y -z y -o {dir_ouput} -f {filename_ouput} '
                   f'{dir_input}')

            if verbose:
                print(f"[{sub_tag}] Creating '{bids_filepath}'...")

            # convert
            if (not os.path.exists(bids_filepath)) or force_conversion:
                # convert to nifti
                if force_conversion:
                    try:
                        os.remove(bids_filepath)
                    except FileNotFoundError:
                        pass  # probably already deleted

                if verbose:
                    print(f"[{sub_tag}] Converting '{dicom_dir}/*' to "
                        f"'{bids_filepath}'...")

                    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

                if name == 'b0_magnitude':

                    fielmap_fpath = os.path.join(
                                    os.path.dirname(bids_filepath),
                                    sub_tag + '_magnitude_fieldmaphz.nii.gz')
                    new_fpath = os.path.join(
                                        os.path.dirname(bids_filepath),
                                        sub_tag + '_fieldmap.nii.gz'
                                        )
                    shutil.move(fielmap_fpath, new_fpath)

                    fielmap_fpath = os.path.join(
                                    os.path.dirname(bids_filepath),
                                    sub_tag + '_magnitude_fieldmaphz.json')
                    new_fpath = os.path.join(
                                        os.path.dirname(bids_filepath),
                                        sub_tag + '_fieldmap.json'
                                        )
                    shutil.move(fielmap_fpath, new_fpath)

            else:
                if verbose:
                    print(f"[{sub_tag}] Skipped!")


def archive_script_code(root_bids_dir):
    """ Archive this script in code/ dir in the produced BIDS valid dataset.
    """
    root_bids_code_dir = os.path.join(root_bids_dir, 'code')

    if not os.path.exists(root_bids_code_dir):
        os.makedirs(root_bids_code_dir)

    shutil.copyfile(__file__, os.path.join(root_bids_code_dir, __file__))


if __name__ == '__main__':

    # ./bidsify_synchropioid.py -v -i /acquistions/SIGNA/ -o /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/ -l dicom_subjects_list.txt --cpu 20

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i','--dicom-root-dir',
                        help='Acquisition DICOM directory',
                        required=True)
    parser.add_argument('-l','--list-subjects',
                        help='DICOM subject folders to include',
                        default='dicom_subjects_list.txt')
    parser.add_argument('-o','--bids-output', help='BIDS root directory',
                        default='synchropioid_data_no_prepro')
    parser.add_argument('-f', '--force-reconversion',
                        help='Force (re-)conversion of the DICOM files',
                        action='store_true', default=False)
    parser.add_argument('--cpu', help='Number of CPU', type=int,
                        default=1)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true',
                        default=False)
    args = vars(parser.parse_args())

    dicom_root_dir = os.path.normpath(args['dicom_root_dir'])
    list_subjects = os.path.normpath(args['list_subjects'])
    root_bids_dir = os.path.normpath(args['bids_output'])
    force_reconversion = args['force_reconversion']
    verbose = args['verbose']

    with open(list_subjects, 'r') as txtfile:
        all_subjects = txtfile.readlines()

    # XXX remove the '\n' character
    all_subjects = [sub[:-1] for sub in all_subjects if sub[-1] == '\n']

    sub_root_dicom_dirs = [os.path.join(dicom_root_dir, subject)
                           for subject in all_subjects]

    archive_script_code(root_bids_dir)

    change_filename = 'CHANGES'
    with open(os.path.join(root_bids_dir, change_filename), 'w') as f:
        f.write(CHANGES_)

    readme_filename = 'README'
    with open(os.path.join(root_bids_dir, readme_filename), 'w') as f:
        f.write(README_)

    descr_filename = 'dataset_description.json'
    with open(os.path.join(root_bids_dir, descr_filename), 'w') as f:
        json.dump(DESCR_, f)

    rest_descr_filename = 'task-rest_bold.json'
    with open(os.path.join(root_bids_dir, rest_descr_filename), 'w') as f:
        json.dump(REST_DESCR_, f)

    hbrest_descr_filename = 'task-hbrest_bold.json'
    with open(os.path.join(root_bids_dir, hbrest_descr_filename), 'w') as f:
        json.dump(HBREST_DESCR_, f)



    def dicomfiles_to_bidsfiles_single_subject(sub_idx,
                                               sub_root_dicom_dir,
                                               root_bids_dir,
                                               verbose=False):
        """ Transform the DICOM files into a correct BIDS directory for a
        single subject."""
        if verbose:
            print(f"[sub-{sub_idx:02d}] found at '{sub_root_dicom_dir}'")

        # get files tree
        sub_dicom_tree = get_sub_dicom_filenames(sub_root_dicom_dir)
        sub_bids_tree = get_sub_bids_filenames(root_bids_dir, sub_idx)

        # convert dicom to nifti files
        transform_dicomfiles_to_bidsfiles(sub_dicom_tree, sub_bids_tree,
                                          force_conversion=force_reconversion,
                                          verbose=verbose)

    # BIDSify all subjects in parallel
    Parallel(n_jobs=args['cpu'], verbose=0)(
        delayed(dicomfiles_to_bidsfiles_single_subject)(
            sub_idx, sub_root_dicom_dir, root_bids_dir, True)
                for sub_idx, sub_root_dicom_dir
                                in enumerate(sub_root_dicom_dirs, start=1))

    # generate participants.tsv file
    subject_tags = pd.DataFrame(columns=['participant_id', 'DICOM tag'])
    for sub_idx, sub_root_dicom_dir in enumerate(sub_root_dicom_dirs, start=1):
        dicom_tag = os.path.basename(os.path.abspath(sub_root_dicom_dir))
        subject_tags.loc[sub_idx] = [f"sub-{sub_idx:02d}", dicom_tag]
    subject_tags_filename = "participants.tsv"
    subject_tags.to_csv(os.path.join(root_bids_dir, subject_tags_filename),
                        sep='\t', index=False)

    # remove .nii, .bval and .bvec files
    subprocess.call(f'find {root_bids_dir} -name "*.nii" -delete',
                    shell=True, stdout=subprocess.DEVNULL)
    subprocess.call(f'find {root_bids_dir} -name "*.bval" -delete',
                    shell=True, stdout=subprocess.DEVNULL)
    subprocess.call(f'find {root_bids_dir} -name "*.bvec" -delete',
                    shell=True, stdout=subprocess.DEVNULL)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Command runs in: {}".format(delta_t))
