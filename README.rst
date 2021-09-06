Synchropioid analysis pipeline
==============================

Preprocesssing step
-------------------

BIDSify the DICOM files
~~~~~~~~~~~~~~~~~~~~~~~

Running **bidsify_synchropioid**::

    ./bidsify_synchropioid -i /home/hcherkaoui/DATA/synchropioid/dicom_dir/ -o ../data/ -v


Fmri-prep preprocess the nifti files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launching **fmriprep** on a laptop::

    sudo nice -n 5 docker run -ti --rm -v /home/hcherkaoui/DATA/synchropioid/nifti_dir/:/data:ro -v /home/hcherkaoui/DATA/synchropioid/nifti_dir/:/derivatives:rw -v /home/hcherkaoui/licenses/license.txt:/opt/freesurfer/license.txt:ro poldracklab/fmriprep:latest /data /derivatives/ participant --output-space MNI152Lin --fs-license-file /opt/freesurfer/license.txt --fs-no-reconall --nthreads 3

Launching **fmriprep** on servers (Drago)::

    nice -n 5 docker run -u 658787 -ti --rm -v /storage/store2/work/hcherkaoui/nifti_dir:/data:ro -v /storage/store2/work/hcherkaoui/nifti_dir/:/derivatives:rw -v /home/parietal/hacherka/license.txt:/opt/freesurfer/license.txt:ro poldracklab/fmriprep:latest /data /derivatives/out participant --fs-license-file /opt/freesurfer/license.txt --output-space MNI152Lin --fs-no-reconall --nthreads 20


Nilearn preprocess the nifti files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    python3 nilearn_preprocessing.py -i nifti_dir/derivatives/ -o preproc_dir --cpu 4 --verbose 1


HRF estimation step
-------------------

Running **decomposition_multi_subjects**::

    python3 decomposition_multi_subjects.py --max-iter 100 --seed 0 --preproc-dir preproc_data --results-dir results_slrda --cpu 4 --verbose 1


Connectome estimation step
--------------------------

Running **estimation_connectome**::

    python3 estimation_connectome.py --preproc-dir preproc_data --result-dir results_connectome --verbose 1


Plotting step
-------------
