""" Plot the feature importance for the decoding on z-maps. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import time
import argparse
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pingouin as pg


# Main
if __name__ == '__main__':

    # python3 plot_anova_connectomes.py --connectomes-dir ../03_connectome/results_connectome/ --plots-dir plots --seed 0 --cpu 3 --task-filter only_hb_rest --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--connectomes-dir', type=str,
                        default='results_connectome',
                        help='Set the name of the connectomes directory.')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Plots directory.')
    parser.add_argument('--cpu', type=int, default=1,
                        help='Set the number of CPU for the decomposition.')
    parser.add_argument('--seed', type=int, default=None, help='Seed.')
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
    # Collect the seed base anlaysis z-maps
    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir, exist_ok=True)

    template_connectome_paths = os.path.join(args.connectomes_dir,
                                             f"corr_matrice_sub-*_run-*_"
                                             f"tr-*_group-*.npy")
    connectome_paths = glob(template_connectome_paths)

    connectome = np.load(connectome_paths[0])
    connectome = connectome[np.tril(connectome, -1) == 0].flatten()
    n_conn_entries = len(connectome)
    columns = ['sub', 'run', 'group', 'conn_entry']
    all_df = [pd.DataFrame(columns=columns)] * n_conn_entries

    for i, connectome_path in enumerate(connectome_paths):

        connectome_path = os.path.normpath(connectome_path)
        connectome_name = os.path.basename(connectome_path)
        chunks = connectome_name.split('_')
        sub, run, t_r, group = chunks[2], chunks[3], chunks[4], chunks[5]

        sub = int(sub.split('-')[-1])
        run = f"Run-{int(run.split('-')[-1])}"
        group = group.split('-')[-1].split('.')[0]
        group = 0 if group == 'control' else 1
        t_r = float(t_r.split('-')[-1])

        connectome = np.load(connectome_path)
        square_shape = connectome.shape
        connectome = connectome[np.tril(connectome, -1) == 0].flatten()

        if t_r in valid_tr:

            if args.verbose:
                print(f"\r[{i+1:02d}/{len(connectome_paths):02d}] Connectome"
                    f" extraction for '{connectome_path}'", end='')

            for j, conn_entry in enumerate(connectome):
                new_col = [sub, run, group, conn_entry]
                all_df[j].loc[len(all_df[j]) + 1] = new_col

    print()

    ###########################################################################
    # Plotting
    l_hypothesis = ['group', 'run * group', 'run']
    nrows, ncols = 1, len(l_hypothesis)
    fig, axis = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 8))

    for hypothesis, ax in zip(l_hypothesis, axis):

        print(f"Generating p-value for hypothesis '{hypothesis}'")

        anova_p_val = []
        for i, df in enumerate(all_df):
            anova_results = pg.rm_anova(dv='conn_entry',
                                        within=['run', 'group'],
                                        subject='sub', data=df, detailed=True)

            filter = anova_results['Source'] == hypothesis
            p_val = anova_results[filter]['p-unc']
            anova_p_val.append(float(p_val))

            if args.verbose:
                print(f"\r[{i+1:02d}/{len(all_df):02d}] ANOVA computation...",
                    end='')

        print()

        anova_p_val = np.nan_to_num(anova_p_val, copy=True)

        anova_log10_p_val = np.log10(anova_p_val)
        anova_log10_p_val_img = anova_log10_p_val.reshape(square_shape)

        sns.heatmap(anova_log10_p_val_img, square=True, vmin=0.,
                    vmax=3., ax=ax, cbar_kws={"shrink": .7})
        ax.set_title(f"Hypothesis: {hypothesis}")

    fig.tight_layout()

    filename = f"anova_connectomes"

    filepath = os.path.join(args.plots_dir, filename + '.pdf')
    print(f"Saving plot at '{filepath}'")
    fig.savefig(filepath, dpi=300)

    filepath = os.path.join(args.plots_dir, filename + '.png')
    print(f"Saving plot at '{filepath}'")
    fig.savefig(filepath, dpi=300)
    fig.tight_layout()

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
