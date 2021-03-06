Synchropioid analysis pipeline
==============================


**A:**
------

To reproduct the results from the analysis of the Synchropioid project, you will need to go through each steps of this tutorial.



0/ Downloading the tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**a/** To download the following tutorial:

- Go to the public repository https://github.com/hcherkaoui/analysis_synchropioid

- Click on the green 'Code' button

- Click on the 'Download ZIP' button

- After downloading the repository, go to your chosen downloading folder and extract the downloaded compressed folder with ::

    unzip analysis_synchropioid-main.zip


- To start the tutorial, go to its root folder: ::

    cd analysis_synchropioid-main


1/ Installation step
~~~~~~~~~~~~~~~~~~~~

**a/** This step installs the tools to perform the analysis.

Install key packages needed for the tutorial::

    sudo apt install python3-pip liblapack-dev docker.io


**b/** Install the needed Python tools with the requirements.txt file::

    python3 -m pip install -r requirements.txt


/!\ If **pip** struggles to install all package, re-run multiple time the command to force pip to resolve the problems.


2/ DICOM files BIDSification step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need steps 0/ and 1/

**a/** Running **generate_symbolic_dicom_dir.py** and **bidsify_synchropioid.py** on 20 CPUs to make the Synchropioid dataset BIDS compliant::

    cd 00_prepro_fmri/
    ./bidsify_synchropioid.py -v -i /biomaps/acquisitions/dicom/SIGNA_PET/ -o /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/ --cpu 20

3/ Nifti files fmri-prep preprocessing step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need steps 0/, 1/ and 2/

**a/** Launching **fmriprep** on a console::

    sudo docker run -ti --rm -v /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/:/data:ro -v /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/:/derivatives:rw -v /biomaps/freesurfer/license.txt:/opt/freesurfer/license.txt:ro poldracklab/fmriprep:latest /data /derivatives/ participant --output-space MNI152Lin --fs-license-file /opt/freesurfer/license.txt --fs-no-reconall --random-seed 0 --dummy-scans 10 --nthreads 20


/!\ If you need to restart a failed preprocessing step, first you will need to delete the previous failed produced folder::

    rm -rf /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/fmriprep/


Simply re-run the docker command after the deleting step.

**b/** To match the BIDS format, rename the produced preprocessing folder ::

    mv /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/fmriprep/ /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/


4/ Quality check step
~~~~~~~~~~~~~~~~~~~~~

Need steps 0/, 1/, 2/ and 3/

**a/** Open the each fmri-prep **.html** report (one per subject) under the folder **/biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/** to check each preprocessing step.


**b/** Run **MRI-quality-control** to add a simple quality check::

    mriqc /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/ . participant


5/ HRF estimation step
~~~~~~~~~~~~~~~~~~~~~~

Need steps 0/, 1/, 2/, 3/ and 4/

**a/** Running **decomposition_multi_subjects** and **decomposition_multi_groups.py** to estimation the HRFs for each subject::

    cd 02_hrf_est/
    python3 decomposition_multi_subjects.py --max-iter 100 --seed 0 --preproc-dir /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/ --results-dir results_hrf_estimation_single --cpu 20 --verbose 1


Those commands will produced the folder **results_hrf_estimation_single** and **results_hrf_estimation_group** that store the produced results.

6/ Connectome estimation step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need steps 0/, 1/, 2/, 3/ and 4/

**a/** Running **estimation_connectome**::

    cd 03_connectome/
    python3 estimation_connectome.py --preproc-dir /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/derivatives/ --result-dir results_connectome --verbose 1


Those commands will produced the folder **results_connectome** that store the produced results.


8/ HRF plotting step
~~~~~~~~~~~~~~~~~~~~

Need steps 0/, 1/, 2/, 3/, 4/, 5/ and 7/

**a/** Produce the silouette curve w.r.t the temporal regularization value plot for the **HRF estimation**::

    cd 05_plotting_hrf_est/
    python3 plot_silhouette_score_per_params_single.py --plots-dir plots --results-dir ../03_hrf_est/results_hrf_estimation_single/ --verbose 1


**b/** Produce vascular maps for each subjects for the best temporal regularization value for the **HRF estimation**::

    python3 haemodynamic_maps_per_subjects.py --bids-root-dir /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/ --results-dir ../03_hrf_est/results_hrf_estimation_single/ --best-params-file decomp_params/best_single_subject_decomp_params.json --output-dir output_dir  --verbose 1


**c/** Produce the haemodynamic region scatter plot and the mean haemodynamic evolution plot for the **HRF estimation**::

    python3 plot_haemodynamic_delays_comparison_subjects.py --plots-dir plots --bids-root-dir /biomaps/synchropioid/dataset_synchropioid/fmri_nifti_dir/ --results-dir ../03_hrf_est/results_hrf_estimation_single/ --best-params-file decomp_params/best_single_subject_decomp_params.json --verbose 1


**d/** Produce the T-test on the vascular maps plot for the **HRF estimation**::

    python3 plot_t_test_per_run.py --vascular-maps-dir output_dir --plots-dir plots --verbose 1
    eog plots/


**e/** Produce the temgesic vs contro examples comparisons on the vascular maps plot for the **HRF estimation**::

    python3 plot_temgesics_vs_control_group.py --vascular-maps-dir output_dir  --bids-root-dir /media/veracrypt1/synchropioid/fmri_nifti_dir/ --plots-dir plots --task-filter only_hb_rest --verbose 1


All the plots are gathered under the **plots** folder.


8/ Connectome plotting step
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need steps 0/, 1/, 2/, 3/, 4/, 6/ and 7/

**a/** Produce the norm plot for the **Connectome**::

    cd 04_plotting_connectome/
    python3 plot_connectome_norm_evolution.py --connectome-dir ../04_connectome/results_connectome/ --plots-dir plots --verbose 1


**b/** Produce the temgesic/control prediction plot for the **Connectome**::

    python3 plot_decoding_connectomes.py --connectomes-dir ../04_connectome/results_connectome/ --plots-dir plots --seed 0 --cpu 20 --verbose 1


**c/** Produce the learning curve for the temgesic/control prediction task plot for the **Connectome**::

    python3 plot_learning_curve_connectomes.py --connectomes-dir ../04_connectome/results_connectome/ --plots-dir plots --seed 0 --cpu 20 --verbose 1


**d/** Produce the T-test on the connectome matrices plot for the **Connectome**::

    python3 plot_t_test_per_run.py --connectome-dir ../04_connectome/results_connectome/ --plots-dir plots --verbose 1
    eog plots/


**e/** Produce the ANOVA on the connectome matrices plot for the **Connectome**::

    python3 plot_anova_connectomes.py --connectomes-dir ../03_connectome/results_connectome/ --plots-dir plots --seed 0 --cpu 20 --task-filter only_hb_rest --verbose 1


All the plots are gathered under the **plots** folder.


**B:**
------

To add a new subject to the Synchropioid dataset, simply edit the ``dicom_subjects_list.txt`` file by adding a newline with the corresponding DICOM directory name (e.g. add a new line ``S00...``).