Synchropioid analysis pipeline
==============================


Main intructions to reproduct the analysis pipeline of the Synchropioid project.


1/ Installation step
~~~~~~~~~~~~~~~~~~~~

Launching the python dependencies installation script::

    python3 -m pip install -r requirements.txt


2/ DICOM files BIDSification step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need steps 1/

Running **bidsify_synchropioid.py**::

    ./bidsify_synchropioid.py -i /biomaps/synchropioid/dataset_synchropioid/fmri_dicom_dir/ -o /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/ -v -n 3


3/ Nifti files fmri-prep preprocessing step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need steps 1/ and 2/

Launching **fmriprep** on a console::

    sudo docker run -ti --rm -v /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/:/data:ro -v /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/:/derivatives:rw -v /home/hcherkaoui/licenses/license.txt:/opt/freesurfer/license.txt:ro poldracklab/fmriprep:latest /data /derivatives/ participant --output-space MNI152Lin --fs-license-file /opt/freesurfer/license.txt --fs-no-reconall --random-seed 0 --dummy-scans 10 --nthreads 20


4/ Quality check step
~~~~~~~~~~~~~~~~~~~~~

Need steps 1/, 2/ and 3/

Open the fmri-prep HTML reports at /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/, then

Run **MRI-quality-control** on a console::

    mriqc /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/ . participant


5/ HRF estimation step
~~~~~~~~~~~~~~~~~~~~~~

Need steps 1/, 2/, 3/ and 4/

Running **decomposition_multi_subjects** and **decomposition_multi_groups.py**::

    cd 03_hrf_est/
    python3 decomposition_multi_subjects.py --max-iter 100 --seed 0 --preproc-dir /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/ --results-dir results_hrf_estimation --cpu 20 --verbose 1

    cd 03_hrf_est/
    python3 decomposition_multi_groups.py --max-iter 100 --seed 0 --preproc-dir /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/ --results-dir results_hrf_estimation --cpu 20 --verbose 1


6/ Connectome estimation step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need steps 1/, 2/, 3/ and 4/

Running **estimation_connectome**::

    cd 04_connectome/
    python3 estimation_connectome.py --preproc-dir /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/ --result-dir results_connectome --verbose 1


7/ Seed-base estimation step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need steps 1/, 2/, 3/ and 4/

Running **seed_base_analysis**::

    cd 05_seed_base_analysis/
    python3 seed_base_analysis.py --preproc-dir /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/ --result-dir results_connectome --verbose 1


8/ Plotting step
~~~~~~~~~~~~~~~~

Need steps 1/, 2/, 3/, 4/, 5/, 6/ and 7/

Plotting **HRF estimation**::

    cd 06_plotting_hrf_est/
    python3 plot_silhouette_score_per_params_single.py --plots-dir plots --results-dir ../03_hrf_est/results_hrf_estimation/ --verbose 1
    python3 plot_haemodynamic_delays_comparison_subjects.py --plots-dir plots --bids-root-dir /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/ --results-dir ../03_hrf_est/results_hrf_estimation/ --best-params-file decomp_params/best_single_subject_decomp_params.json --verbose 1
    python3 haemodynamic_maps_per_subjects.py --bids-root-dir /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/ --results-dir ../03_hrf_est/results_hrf_estimation/ --best-params-file decomp_params/best_single_subject_decomp_params.json --output-dir output_dir  --verbose 1
    python3 plot_t_test_per_run.py --vascular-maps-dir output_dir --plots-dir plots --verbose 1


Plotting **Connectome**::

    cd 07_plotting_connectome/
    python3 plot_connectome_norm_evolution.py --connectome-dir ../04_connectome/results_connectome/ --plots-dir plots --verbose 1
    python3 plot_decoding_connectomes.py --connectomes-dir ../04_connectome/results_connectome/ --plots-dir plots --seed 0 --cpu 3 --verbose 1
    python3 plot_learning_curve_connectomes.py --connectomes-dir ../04_connectome/results_connectome/ --plots-dir plots --seed 0 --cpu 3 --verbose 1
    python3 plot_t_test_per_run.py --connectome-dir ../04_connectome/results_connectome/ --plots-dir plots --verbose 1


Plotting **Seed base analysis**::

    cd 08_plotting_seed_base_analysis/
    python3 decoding_z_maps.py --z-maps-dir ../05_seed_base_analysis/z_maps/ --plots-dir plots --seed 0 --cpu 3 --verbose 1
    python3 plot_mean_z_maps.py --z-maps-dir ../05_seed_base_analysis/z_maps/ --plots-dir plots --verbose 1
    python3 t_test_per_run.py --z-maps-dir output_dir --plots-dir plots --verbose 1

