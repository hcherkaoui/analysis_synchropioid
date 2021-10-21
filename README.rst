Synchropioid analysis pipeline
==============================

Preprocesssing step
-------------------

DICOM files BIDSification step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running **bidsify_synchropioid.py**::

    ./bidsify_synchropioid.py -i /media/veracrypt1/synchropioid/dicom_dir/ -o /media/veracrypt1/synchropioid/nifti_dir/ -v -n 3

Nifti files fmri-prep preprocessing step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launching **fmriprep** on a console::

    sudo nice -n 5 docker run -ti --rm -v /media/veracrypt1/synchropioid/nifti_dir/:/data:ro -v /media/veracrypt1/synchropioid/nifti_dir/:/derivatives:rw -v /home/hcherkaoui/licenses/license.txt:/opt/freesurfer/license.txt:ro poldracklab/fmriprep:latest /data /derivatives/ participant --output-space MNI152Lin --fs-license-file /opt/freesurfer/license.txt --fs-no-reconall --random-seed 0 --nthreads 20

HRF estimation step
-------------------

Running **decomposition_multi_subjects**::

    python3 decomposition_multi_subjects.py --max-iter 100 --seed 0 --preproc-dir /media/veracrypt1/synchropioid/nifti_dir/derivatives/ --results-dir results_slrda --cpu 4 --verbose 1


Connectome estimation step
--------------------------

Running **estimation_connectome**::

    python3 estimation_connectome.py --preproc-dir /media/veracrypt1/synchropioid/nifti_dir/derivatives/ --result-dir results_connectome --verbose 1


Plotting step
-------------
