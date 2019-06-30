""" Simple GLM analysis of the synchropioid data.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import subprocess  # XXX hack to concatenate the spatial maps in one pdf
import shutil
from datetime import datetime
import bids
import numpy as np
import matplotlib.pylab as plt
from nilearn import image, plotting
from nilearn.decomposition import CanICA


from pprint import pprint

###############################################################################
# Global functions and variables
TR = 2.0

def join_dirs(l_paths):
    """ Joined all paths together. """
    if len(l_paths) == 1:
        return l_paths
    joined_path = l_paths[0]
    for path in l_paths[1:]:
        joined_path = os.path.join(joined_path, path)

    return joined_path

###############################################################################
# Load data
synchropioid_dataset_path = join_dirs([os.path.expanduser('~'), 'DATA',
                                      'synchropioid_data'])
layout = bids.BIDSLayout(synchropioid_dataset_path, derivatives=True)
print("Find Synchropioid dataset at '{}'".format(layout))

###############################################################################
# Saving management
date = datetime.now()
root_dir = 'results_ica_#{0}{1}{2}{3}{4}{5}'.format(date.year, date.month,
                                                    date.day, date.hour,
                                                    date.minute, date.second)
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

shutil.copyfile(__file__, os.path.join(root_dir, __file__))

components_dir = os.path.join(root_dir, 'components')
if not os.path.exists(components_dir):
    os.makedirs(components_dir)

plots_dir = os.path.join(root_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

###############################################################################
# Main
for idx_sub in ['02', '03', '04']:

    print("Subject-{0}:".format(idx_sub))

    derivative_dir = join_dirs([synchropioid_dataset_path, 'derivatives',
                            'fmriprep', 'sub-{}'.format(idx_sub), 'func'])
    func_files = layout.get(scope='derivatives', subject=idx_sub,
                            task='rest', suffix='preproc')
    func_files = [os.path.join(derivative_dir, func_file.filename)
                  for func_file in func_files]

    canica = CanICA(n_components=20, smoothing_fwhm=6.,
                    memory=".cache", memory_level=2,
                    threshold=3., verbose=10, random_state=0)
    canica.fit(sorted(func_files))

    components_img = canica.components_img_
    fname = 'canica_sub-{}.nii.gz'.format(idx_sub)
    fname = os.path.join(components_dir, fname)
    components_img.to_filename(fname)

    plotting.plot_prob_atlas(components_img, title='All ICA components')
    fname = 'canica_sub-{0}_all_comp.pdf'.format(idx_sub)
    fname = os.path.join(plots_dir, fname)
    plt.savefig(fname, dpi=150)

    for i, cur_img in enumerate(image.iter_img(components_img)):
        plotting.plot_stat_map(cur_img, title="IC %d" % i, colorbar=True)
        fname = 'canica_sub-{0}_comp_{1:02d}.pdf'.format(idx_sub, i)
        fname = os.path.join(plots_dir, fname)
        plt.savefig(fname, dpi=150)

    pattern_files = 'canica_sub-{0}_comp_*.pdf'.format(idx_sub)
    pdf_files = os.path.join(plots_dir, pattern_files)
    final_file = 'canica_sub-{0}_all_comp_sep.pdf'.format(idx_sub)
    pdf_file = os.path.join(plots_dir, final_file)
    subprocess.call("pdftk {} cat output {}".format(pdf_files, pdf_file),
                    shell=True)
    subprocess.call("rm -f {}".format(pdf_files), shell=True)
