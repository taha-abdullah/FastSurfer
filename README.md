[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deep-MI/FastSurfer/blob/stable/Tutorial/Tutorial_FastSurferCNN_QuickSeg.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deep-MI/FastSurfer/blob/stable/Tutorial/Complete_FastSurfer_Tutorial.ipynb)

# Overview

This directory contains all information needed to run FastSurfer - a fast and accurate deep-learning based neuroimaging pipeline. This approach provides a full [FreeSurfer](https://freesurfer.net/) alternative for volumetric analysis (within 1 minute) and surface-based thickness analysis (within only around 1h run time). The whole pipeline consists of two main parts:

(i) FastSurferCNN - an advanced deep learning architecture capable of whole brain segmentation into 95 classes in under
1 minute, mimicking FreeSurfer’s anatomical segmentation and cortical parcellation (DKTatlas)

(ii) recon-surf - full FreeSurfer alternative for cortical surface reconstruction, mapping of cortical labels and traditional point-wise and ROI thickness analysis in approximately 60 minutes. For surface group analysis, sphere.reg can be additionally generated by adding the --surfreg flag.

Image input requirements are identical to FreeSurfer: good quality T1-weighted MRI acquired at 3T with a resolution close to 1mm isotropic (slice thickness should not exceed 1.5mm). Preferred sequence is Siemens MPRAGE or multi-echo MPRAGE. GE SPGR should also work. Sub-mm scans (e.g. .75 or .8mm isotropic) will be downsampled by us automatically to 1mm isotropic, for example, we had success segmenting de-faced HCP data.

Within this repository, we provide the code and Docker files for running FastSurferCNN (segmentation only) and recon-surf (surface pipeline only) independently from each other or as a whole pipeline (run_fastsurfer.sh, segmentation + surface pipeline). For each of these purposes, see the README.md's in the corresponding folders.

![](/images/teaser.png)

## Usage
There are three ways to run FastSurfer - (a) using Docker, (b) using Singularity, (c) as a native install. The recommended way is to use Docker or Singularity. If either is already installed on your system, there are only two commands to get you started (the download of a container image, and running it). Installation instructions, especially for the more involved native install, can be found in [INSTALL](INSTALL.md).

(a) When using __docker__, simply pull our official images from [Dockerhub](https://hub.docker.com/r/deepmi/fastsurfer) or
use the provided Dockerfiles in our Docker directory to build your own image (see the [README](docker/README.md) in the Docker directory). No other local installations are needed (FreeSurfer and everything else will be included, you only need to provide a [FreeSurfer license file](https://surfer.nmr.mgh.harvard.edu/fswiki/License)). See [Example 3](#example-3:-fastSurfer-inside-docker) for an example how to run FastSurfer inside a Docker container. Mac users need to increase docker memory to 15 GB by overwriting the settings under Docker Desktop --> Preferences --> Resources --> Advanced (slide the bar under Memory to 15 GB; see: [docker for mac](https://docs.docker.com/docker-for-mac/) for details).

(b) When using __singularity__, use the desired Docker image to build the corresponding Singularity image (see the [README](singularity/README.md) in the Singularity directory for directions on building your own Singularity images from Docker). Note that the `--gpus all` argument from Docker is replaced with `--nv` when running in Singularity, along with other arguments shown in the example command below (see [Example 4](#example-4:-fastSurfer-inside-singularity)).

(c) For a __native install__ on a modern linux (we tested Ubuntu 20.04), download this github repository (use git clone or download as zip and unpack) for the necessary source code and python scripts. You also need to have the necessary Python 3 libraries installed (see __requirements.txt__) as well as bash-4.0 or higher (when using pip, upgrade pip first as older versions will fail). This is already enough to generate the whole-brain segmentation using FastSurferCNN (see the README.md in the FastSurferCNN directory for the exact commands). In order to run the whole FastSurfer pipeline (for surface creation etc.) locally on your machine, a working version of __FreeSurfer__ (v7.2, [https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads)) is required to be pre-installed and sourced. See [INSTALL](INSTALL.md) for detailled installation instructions and [Example 1](#example-1:-fastSurfer-on-subject1) or [Example 2](#example-2:-fastSurfer-on-multiple-subjects-(parallel-processing)) for an illustration of the commands to run the entire FastSurfer pipeline (FastSurferCNN + recon-surf) natively.

In the native install, the main script called __run_fastsurfer.sh__ can be used to run both FastSurferCNN and recon-surf sequentially on a given subject. There are several options which can be selected and set via the command line. 
List them by running the following command:
```bash
./run_fastsurfer.sh --help
```

### Required arguments
* --sd: Output directory \$SUBJECTS_DIR (equivalent to FreeSurfer setup --> $SUBJECTS_DIR/sid/mri; $SUBJECTS_DIR/sid/surf ... will be created).
* --sid: Subject ID for directory inside \$SUBJECTS_DIR to be created ($SUBJECTS_DIR/sid/...)
* --t1: T1 full head input (not bias corrected, global path). The network was trained with conformed images (UCHAR, 256x256x256, 1 mm voxels and standard slice orientation). These specifications are checked in the eval.py script and the image is automatically conformed if it does not comply.

### Required for docker
* --fs_license: Path to FreeSurfer license key file. Register (for free) at https://surfer.nmr.mgh.harvard.edu/registration.html to obtain it if you do not have FreeSurfer installed so far. Strictly necessary if you use Docker, optional for local install (your local FreeSurfer license will automatically be used).

### Network specific arguments (optional)
* --seg: Global path with filename of segmentation (where and under which name to store it). The file can be in mgz, nii, or nii.gz format. Default location: $SUBJECTS_DIR/$sid/mri/aparc.DKTatlas+aseg.deep.mgz
* --weights_sag: Pretrained weights of sagittal network. Optional, will download checkpoint if missing.
* --weights_ax: Pretrained weights of axial network. Optional, will download checkpoint if missing.
* --weights_cor: Pretrained weights of coronal network. Optional, will download checkpoint if missing.
* --seg_log: Name and location for the log-file for the segmentation (FastSurferCNN). Default: $SUBJECTS_DIR/$sid/scripts/deep-seg.log
* --clean_seg: Flag to clean up NN segmentation
* --run_viewagg_on: Define where the view aggregation should be run on. 
                    By default, the program checks if you have enough memory to run the view aggregation on the gpu. 
                    The total memory is considered for this decision. 
                    If this fails, or you actively overwrote the check with setting "--run_viewagg_on cpu", view agg is run on the cpu. 
                    Equivalently, if you define "--run_viewagg_on gpu", view agg will be run on the gpu (no memory check will be done).
* --device <str>: Select device for NN segmentation ("cpu", "cuda"), where cuda (default) means Nvidia GPU, you can select which one "cuda:1".
* --batch: Batch size for inference. Default: 8. Lower this to reduce memory requirement
* --order: Order of interpolation for mri_convert T1 before segmentation (0=nearest, 1=linear(default), 2=quadratic, 3=cubic)

### Surface pipeline arguments (optional)
* --fstess: Use mri_tesselate instead of marching cube (default) for surface creation
* --fsqsphere: Use FreeSurfer default instead of novel spectral spherical projection for qsphere
* --fsaparc: Use FS aparc segmentations in addition to DL prediction (slower in this case and usually the mapped ones from the DL prediction are fine)
* --surfreg: Create Surface-Atlas (sphere.reg) registration with FreeSurfer (for cross-subject correspondence or other mappings)
* --parallel: Run both hemispheres in parallel
* --threads: Set openMP and ITK threads to <int>

### Other
* --py: which python version to use. Default: python3.8
* --seg_only: only run FastSurferCNN (generate segmentation, do not run the surface pipeline)
* --surf_only: only run the surface pipeline recon_surf. The segmentation created by FastSurferCNN must already exist in this case.
    

### Example 1: Native FastSurfer on subject1 (with parallel processing of hemis)

Given you want to analyze data for subject1 which is stored on your computer under /home/user/my_mri_data/subject1/orig.mgz, run the following command from the console (do not forget to source FreeSurfer!):

```bash
# Source FreeSurfer
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
datadir=/home/user/my_mri_data
fastsurferdir=/home/user/my_fastsurfer_analysis

# Run FastSurfer
./run_fastsurfer.sh --t1 $datadir/subject1/orig.mgz \
                    --sid subject1 --sd $fastsurferdir \
                    --parallel --threads 4
```

The output will be stored in the $fastsurferdir (including the aparc.DKTatlas+aseg.deep.mgz segmentation under $fastsurferdir/subject1/mri (default location)). Processing of the hemispheres will be run in parallel (--parallel flag). Omit this flag to run the processing sequentially.

### Example 2: Native FastSurfer on multiple subjects

In order to run FastSurfer on multiple cases which are stored in the same directory, prepare a subjects_list.txt file listing the names line per line:
subject1\n
subject2\n
subject3\n
...
subject10\n

And invoke the following command (make sure you have enough ressources to run the given number of subjects in parallel!):

```bash
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

cd /home/user/FastSurfer
datadir=/home/user/my_mri_data
fastsurferdir=/home/user/my_fastsurfer_analysis
mkdir -p $fastsurferdir/logs # create log dir for storing nohup output log (optional)

while read p ; do
  echo $p
  nohup ./run_fastsurfer.sh --t1 $datadir/$p/orig.mgz 
                            --sid $p --sd $fastsurferdir > $fastsurferdir/logs/out-${p}.log &
  sleep 90s 
done < ./data/subjects_list.txt
```

### Example 3: FastSurfer Docker
After pulling one of our images from Dockerhub, you do not need to have a separate installation of FreeSurfer on your computer (it is already included in the Docker image). However, if you want to run more than just the segmentation CNN, you need to register at the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license for free. This license needs to be passed to the script via the --fs_license flag. Basically for Docker (as for Singularity below) you are starting a container image (with the run command) and pass several parameters for that, e.g. if GPUs will be used and mouting (linking) the input and output directories to inside the container image. In the second half of that call you pass paramters to our run_fastsurfer.sh script that runs inside the container (e.g. where to find the FreeSurfer license file, and the input data and other flags). 

To run FastSurfer on a given subject using the provided GPU-Docker, execute the following command:

```bash
docker pull deepmi/fastsurfer
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --rm --user XXXX deepmi/fastsurfer \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subject2/orig.mgz \
                      --sid subject2 --sd /output \
                      --parallel
```

Docker flags:
* The --gpus flag is used to allow Docker to access GPU resources. With it you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set --gpus device=0. To use multiple specific ones (e.g. GPU 0, 1 and 3), set --gpus '"device=0,1,3"'.
* The -v commands mount your data, output, and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license). 
* The --rm flag takes care of removing the container once the analysis finished. 
* The --user XXXX part should be changed to the appropriate user id (a four-digit number; can be checked with the command "id -u" on linux systems). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root.

FastSurfer Flags to pass through:
* The fs_license points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 
* Note, that the paths following --fs_license, --t1, and --sd are inside the container, not global paths on your system, so they should point to the places where you mapped these paths above with the -v arguments. 
* A directory with the name as specified in --sid (here subject2) will be created in the output directory. So in this example output will be written to /home/user/my_fastsurfer_analysis/subject2/ . Make sure the output directory is empty, to avoid overwriting existing files. 

You can also run a CPU-Docker with very similar commands. See [Docker/README.md](Docker/README.md) for more details.

### Example 4: FastSurfer Singularity
After building the Singularity image (see instructions in ./Singularity/README.md), you also need to register at the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license (for free) - just as when using Docker. This license needs to be passed to the script via the --fs_license flag.

To run FastSurfer on a given subject using the Singularity image with GPU access, execute the following commands from a directory where you want to store singularity images. This will create a singularity image from our Dockerhub image and execute it:

```bash
singularity build fastsurfer-gpu.sif docker://deepmi/fastsurfer
singularity exec --nv -B /home/user/my_mri_data:/data \
                      -B /home/user/my_fastsurfer_analysis:/output \
                      -B /home/user/my_fs_license_dir:/fs_license \
                       ./fastsurfer-gpu.sif \
                       /fastsurfer/run_fastsurfer.sh \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subject2/orig.mgz \
                      --sid subject2 --sd /output \
                      --parallel
```

Singularity Flags:
* The `--nv` flag is used to access GPU resources. 
* The -B commands mount your data, output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license). 

FastSurfer Flags to pass through:
* The fs_license points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 
* Note, that the paths following --fs_license, --t1, and --sd are inside the container, not global paths on your system, so they should point to the places where you mapped these paths above with the -B arguments. 
* A directory with the name as specified in --sid (here subject2) will be created in the output directory. So in this example output will be written to /home/user/my_fastsurfer_analysis/subject2/ . Make sure the output directory is empty, to avoid overwriting existing files. 

You can run the Singularity equivalent of CPU-Docker by building a Singularity image from the CPU-Docker image and excluding the `--nv` argument in your Singularity exec command.

## System Requirements

Recommendation: At least 8GB CPU RAM and 8GB NVIDIA GPU RAM ```--batch 1 --run_viewagg_on gpu```  

Minimum: 8 GB CPU RAM and 2 GB GPU RAM ```--batch 1 --run_viewagg_on cpu```

CPU-only: 8 GB CPU RAM (much slower, not recommended) ```--device cpu --batch 4``` 

## FreeSurfer Downstream Modules

FreeSurfer provides several Add-on modules for downstream processing, such as subfield segmentation ( [hippocampus/amygdala](https://surfer.nmr.mgh.harvard.edu/fswiki/HippocampalSubfieldsAndNucleiOfAmygdala), [brainstrem](https://surfer.nmr.mgh.harvard.edu/fswiki/BrainstemSubstructures), [thalamus](https://freesurfer.net/fswiki/ThalamicNuclei) and [hypothalamus](https://surfer.nmr.mgh.harvard.edu/fswiki/HypothalamicSubunits) ) as well as [TRACULA](https://surfer.nmr.mgh.harvard.edu/fswiki/Tracula). We now provide symlinks to the required files, as FastSurfer creates them with a different name (e.g. using "mapped" or "DKT" to make clear that these file are from our segmentation using the DKT Atlas protocol, and mapped to the surface). Most subfield segmentations require `wmparc.mgz` and work very well with FastSurfer,  so feel free to run those pipelines after FastSurfer. TRACULA requires `aparc+aseg.mgz` which we now link, but have not tested if it works, given that [DKT-atlas](https://mindboggle.readthedocs.io/en/latest/labels.html) merged a few labels. You should source FreeSurfer 7.2 to run these modules. 

## Intended Use

This software can be used to compute statistics from an MR image for research purposes. Estimates can be used to aggregate population data, compare groups etc. The data should not be used for clinical decision support in individual cases and, therefore, does not benefit the individual patient. Be aware that for a single image, produced results may be unreliable (e.g. due to head motion, imaging artefacts, processing errors etc). We always recommend to perform visual quality checks on your data, as also your MR-sequence may differ from the ones that we tested. No contributor shall be liable to any damages, see also our software [LICENSE](LICENSE). 

## References

If you use this for research publications, please cite:

Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer - A fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219 (2020), 117012. https://doi.org/10.1016/j.neuroimage.2020.117012

Henschel L*, Kügler D*, Reuter M. (*co-first). FastSurferVINN: Building Resolution-Independence into Deep Learning Segmentation Methods - A Solution for HighRes Brain MRI. NeuroImage 251 (2022), 118933. http://dx.doi.org/10.1016/j.neuroimage.2022.118933

Stay tuned for updates and follow us on Twitter: https://twitter.com/deepmilab

## Acknowledgements
The recon-surf pipeline is largely based on FreeSurfer 
https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferMethodsCitation
