#!/usr/bin/env python3
""" Tool to generate a folder with symbolic link to the DICOM folders.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import argparse


if __name__ == '__main__':

    # ./generate_symbolic_dicom_dir.py -v -d /acquistions/SIGMA/ -o /biomaps/synchropioid/dataset_synchropioid/fmri_dicom_dir/ -l dicom_subjects_list.txt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d','--dicom-root-dir',
                        help='Acquisition DICOM directory',
                        required=True)
    parser.add_argument('-o','--dicom-symbolic-dir',
                        help='DICOM symbolic directory',
                        required=True)
    parser.add_argument('-l','--list-subjects',
                        help='DICOM subject folders to include',
                        default='dicom_subjects_list.txt')
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true',
                        default=False)
    args = vars(parser.parse_args())

    if not os.path.exists(args['dicom_symbolic_dir']):
        os.makedirs(args['dicom_symbolic_dir'])

    with open(args['list_subjects'], 'r') as txtfile:
        all_subjects = txtfile.readlines()

    # XXX remove the '\n' character
    all_subjects = [sub[:-1] for sub in all_subjects if sub[-1] == '\n']

    for sub in all_subjects:

        sub_path = os.path.join(args['dicom_root_dir'], sub)
        os.symlink(sub_path, os.path.join(args['dicom_symbolic_dir'], sub))

        if args['verbose']:
            print(f"Creating symbolic link for '{sub}' under "
                  f"'{args['dicom_root_dir']}'")
