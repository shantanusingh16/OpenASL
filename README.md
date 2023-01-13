# OpenASL: A Large-Scale Open-Domain Sign Language Translation Dataset
![openasl-data](assets/teaser.gif)

## Introduction
This repo contains OpenASL dataset proposed in paper: [Open-Domain Sign Language Translation Learned from Online Video](https://arxiv.org/abs/2205.12870)

If you use OpenASL data in your research, please use the following BibTeX entry for citation.

```BibTeX
@inproceedings{shi2022open,
  author = {Bowen Shi and Diane Brentari and Greg Shakhnarovich and Karen Livescu},
  title = {Open-Domain Sign Language Translation Learned from Online Video},
  booktitle = {EMNLP},
  year = {2022}
}
```

## Instructions
### 1. Download videos
Run the following command to download videos into `/path/to/raw-video`.
```
python prep/download.py --tsv data/openasl-v1.0.tsv --dest /path/to/raw-video
```

If you are on slurm, you can parallelize the video downloading with the following command:

``` 
python prep/download.py --tsv data/openasl-v1.0.tsv --dest /path/to/raw-video --nshard ${nshard} --slurm
```

where `${nshard}` is the number of jobs to launch. The slurm arguments can be modified via `--slurm-argument '{"slurm_array_parallelism":100,"slurm_partition":"cpu","timeout_min":240,"slurm_mem":"16g"}'`. Note some videos may no longer be publicly available. 

### 2. Video preprocessing
The following command will trim the raw video into clips and spatially crop the video clips.


```
python prep/process_openasl.py --tsv data/openasl-v1.0.tsv --bbox data/bbox-v1.0.json --raw /path/to/raw-video --output /path/to/output-dir --ffmpeg /path/to/ffmpeg
```

The processed frames will be saved into individual directories in `/path/to/output-dir`. `data/bbox-v1.0.json` contains the person bounding box of each video clip in the format of `[x0,y0,x1,y1]` normalized by the image width and height. Note the bounding boxes are generated based on face detection plus some heuristic processing and are not proofread by humans. Roughly the error rate is below `1%`.

This code differs from crop_video.py only in saving the frames as png and not concatenating them using ffmpeg and saving them as video.

NOTE: The experiments in parallelizing cropping videos using Process Pool all failed due to memory issues with parallelizing FFMPEG. FFMPEG itself uses multithreading and hence can lead to high CPU/MEM loads when used in multiple processes parallely.
The script `crop_video_parallel.py` was an experiment in this direction and SHOULD NOT BE USED.

### 3. Keypoint extraction
The following command will trim the raw video into clips and spatially crop the video clips.


```
python prep/extract_keypts.py --raw /path/to/cropped-frames
```

The processed numpy files will be saved into `OUTPUT_KEYPTS_DIR` set within the file. 
. It will contain 3 sub-directories `body`, `hand` and `face` and each of these will have numpy files with filenames corresponding to the directory names in the provided `raw` path. The numpy files contain the extracted keypoints.


## License
OpenASL is licensed under the Creative Commons BY-NC-ND 4.0 License - see [LICENSE](LICENSE.md) for more details. 
