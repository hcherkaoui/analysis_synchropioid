#!/usr/bin/env python3
""" Tool to find all empty files under 'root_dir'.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import argparse
import pprint
from progress.bar import Bar


def _find_empty_files(root_dir, verbose=False):
    """ Return all the empty DICOM files found under 'root_dir'.
    """
    empty_files = []
    for root, _, fnames in os.walk(root_dir):

        n_empty_files = len(empty_files)
        progr_bar = Bar('Searching', max=len(fnames))

        for fname in fnames:
            ext = fname.split('.')[-1:]
            if '.'.join(ext) == 'dcm':
                fpath = os.path.join(root, fname)
                if  os.path.getsize(fpath) == 0:
                    empty_files.append(os.path.join(root, fpath))
            progr_bar.next()

        if verbose:
            emplty_line = "" if len(fnames) == 0 else "\n" 
            print(f"{emplty_line}    {n_empty_files - len(empty_files)}"
                f" empty DICOM files found under '{root}'")

    return empty_files


if __name__ == '__main__':

    # ./find_empty_dicom.py -v -i DICOM_DIR_TO_CHECK

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i','--root-dir', help='Root directories',
                        required=True)
    parser.add_argument('-v','--verbose', help='Verbose level',
                        action='store_true', default=False)
    args = vars(parser.parse_args())

    if args['verbose']:
        print(f"Searching empty DICOM under '{args['root_dir']}'")

    empty_files = _find_empty_files(args['root_dir'], verbose=args['verbose'])

    if args['verbose']:
        print("#" * 80)
        if len(empty_files) == 0:
            print("No Empty DICOM found.")
        else:
            print("Empty DICOM found:")
            pprint.pprint(empty_files)
    else:
        pprint.pprint(empty_files)
