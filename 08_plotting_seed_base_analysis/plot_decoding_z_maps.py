""" Plot the feature importance for the decoding on z-maps. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import time
import argparse
from glob import glob
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from nilearn import input_data, plotting
from hemolearn import atlas


# set fontsize
sns.set(font_scale=0.85)


def threshold(x, th='10%'):
    x_flatten = x.flatten()
    th = float(th[:-1]) / 100.
    return np.sort(x_flatten)[-int(th * len(x_flatten))]


def add_entry_to_dict(d, k, v):
    if k not in d:
        d[k] = [v]
    else:
        d[k].append(v)
    return d


# Main
if __name__ == '__main__':

    # python3 decoding_z_maps.py --z-maps-dir ../05_seed_base_analysis/z_maps/ --plots-dir plots --seed 0 --cpu 3 --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--z-maps-dir', type=str, default='z_maps_dir',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Plots directory.')
    parser.add_argument('--cpu', type=int, default=1,
                        help='Set the number of CPU for the decomposition.')
    parser.add_argument('--seed', type=int, default=None, help='Seed.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    ###########################################################################
    # Collect the seed base anlaysis z-maps
    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir, exist_ok=True)

    template_z_maps_paths = os.path.join(args.z_maps_dir,
                                         f"sub-*_run-*_group-*_tr-*_z_map"
                                         f".nii.gz")
    z_maps_paths = glob(template_z_maps_paths)

    brain_mask, _ = atlas.fetch_aal3_vascular_atlas()
    masker = input_data.NiftiMasker(smoothing_fwhm=10., mask_img=brain_mask)
    masker.fit(z_maps_paths)

    all_z_maps, all_groups, keys = dict(), dict(), []
    for i, z_maps_path in enumerate(z_maps_paths):

        z_maps_path = os.path.normpath(z_maps_path)

        z_maps_name = os.path.basename(z_maps_path)
        sub, run, group, t_r, _, _ = z_maps_name.split('_')

        sub = int(sub.split('-')[-1])
        run = f"Run-{int(run.split('-')[-1])}"
        group = 0 if group.split('-')[-1] == 'control' else 1
        t_r = f"TR={float(t_r.split('-')[-1]):.1f}s"

        flatten_z_map = masker.transform_single_imgs(z_maps_path).flatten()

        if args.verbose:
            print(f"\r[{i+1:02d}/{len(z_maps_paths):02d}] Formated data saved "
                f"under '{z_maps_path}'", end='')

        key = (run, t_r)
        keys.append(key)
        add_entry_to_dict(all_z_maps, key, flatten_z_map)
        add_entry_to_dict(all_groups, key, group)

        if run in ["Run-3", "Run-5"]:
            key = (run, "TR=all-TR")
            keys.append(key)
            add_entry_to_dict(all_z_maps, key, flatten_z_map)
            add_entry_to_dict(all_groups, key, group)

        key = ("Run-all-runs", t_r)
        keys.append(key)
        add_entry_to_dict(all_z_maps, key, flatten_z_map)
        add_entry_to_dict(all_groups, key, group)

        key = ("Run-all-runs", "TR=all-TR")
        keys.append(key)
        add_entry_to_dict(all_z_maps, key, flatten_z_map)
        add_entry_to_dict(all_groups, key, group)

    print()

    unique_keys = set(keys)

    ###########################################################################
    # Fit the decoder model
    i = 0
    scoring = 'accuracy'
    scores = pd.DataFrame(columns=['run', 't_r', 'score'])
    best_mean_test_scores, best_std_test_score, runs = [], [], []
    for key in unique_keys:

        run, t_r = key
        runs.append(run)

        X = np.c_[all_z_maps[key]]
        y = np.array(all_groups[key])

        if args.verbose:
            print(f"[{run}|{t_r}] Fitting decoder...")

        grid = GridSearchCV(
                    estimator=LogisticRegression(),
                    param_grid=[dict(C=list(np.logspace(-5, 5, 11)),
                                     penalty=['l2', 'l1'],
                                     solver=['liblinear'],
                                     fit_intercept=[False, True])],
                    cv=LeaveOneOut(),
                    scoring=scoring,
                    n_jobs=args.cpu,
                                ).fit(X, y)

        idx = grid.best_index_
        if args.verbose:
            mean_score = grid.cv_results_['mean_test_score'][grid.best_index_]
            std_scsore = grid.cv_results_['std_test_score'][grid.best_index_]
            print(f"[{run}|{t_r}] Best score = {mean_score:.2f} "
                  f"+/- {std_scsore / 2.:.2f}")

        for score in grid.cv_results_['mean_test_score']:
            scores.loc[i] = [run, t_r, score]
            i += 1

        if args.verbose:
            print(f"[{run}|{t_r}] Best parameters: "
                  f"{pprint.pformat(grid.best_params_)}")

        if run == "Run-all-runs" and t_r == "TR=all-TR":

            coef = LogisticRegression(**grid.best_params_).fit(X, y).coef_
            coef_img = masker.inverse_transform(coef)

            t = threshold(coef, th='10%')

            filename = os.path.join(args.plots_dir,
                    f"{scoring}_coef_run-{run}_tr-{t_r}_seed-{args.seed}")
            if args.verbose:
                print(f"Saving plot under '{filename + '.pdf'}'")
            plotting.plot_stat_map(
                            coef_img, colorbar=True, display_mode='z',
                            threshold=t, cut_coords=np.linspace(-50, 70, 8),
                            title=run, output_file=filename + '.pdf')

            if args.verbose:
                print(f"Saving plot under '{filename + '.png'}'")
            plotting.plot_stat_map(
                            coef_img, colorbar=True, display_mode='z',
                            threshold=t, cut_coords=np.linspace(-50, 70, 8),
                            title=run, output_file=filename + '.png')

    ###########################################################################
    # Plotting
    palette = {"TR=2.0s":'tab:blue',
               "TR=0.8s":'tab:orange',
               "TR=all-TR":'tab:red'}
    order = ["Run-1", "Run-2", "Run-3", "Run-4", "Run-5", "Run-all-runs"]

    plt.figure('Boxplot', figsize=(6, 2))
    ax = sns.boxplot(data=scores, x='run', y='score', hue='t_r',
                     palette=palette, order=order, linewidth=0.75,
                     width=0.75)
    for y in [0.0, 0.5, 1.0]:
        plt.axhline(y, lw=.5, color='gray')
    for x in 0.5 + np.arange(len(order)):
        plt.axvline(x, lw=.5, color='gray')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.tight_layout()
    filename = os.path.join(args.plots_dir,
                            f"{scoring}_boxplot_seed-{args.seed}")

    if args.verbose:
        print(f"Saving plot under '{filename + '.pdf'}'")
    plt.savefig(filename + '.pdf', dpi=300)

    if args.verbose:
        print(f"Saving plot under '{filename + '.png'}'")
    plt.savefig(filename + '.png', dpi=300)

    plt.figure('Point cloud', figsize=(6, 2))
    ax = sns.stripplot(x='run', y='score', hue='t_r', data=scores,
                       palette=palette, jitter=0.3, size=2.0, order=order,)
                    #    alpha=.5)
    for y in [0.0, 0.5, 1.0]:
        plt.axhline(y, lw=.75, color='gray')
    for x in 0.5 + np.arange(len(order)):
        plt.axvline(x, lw=.5, color='gray')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.tight_layout()
    filename = os.path.join(args.plots_dir,
                            f"{scoring}_strip_seed-{args.seed}")

    if args.verbose:
        print(f"Saving plot under '{filename + '.pdf'}'")
    plt.savefig(filename + '.pdf', dpi=300)

    if args.verbose:
        print(f"Saving plot under '{filename + '.png'}'")
    plt.savefig(filename + '.png', dpi=300)

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
