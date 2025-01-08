# Visual Multi-object Tracking Algorithms

This is the official Python implementation repository for a paper entitled "Adaptive Confidence Threshold for ByteTrack in Multi-Object Tracking", *ICCAIS 2023 ([arXiv 2312.01650](https://arxiv.org/abs/2312.01650))*.

    - BYTETrack
    - OC-SORT
    - FairMOT
    - SORT
    - CMC (Camera Motion Compensation) 
    - HOTA Evaluation

### Usage
1. **Set Up Python Environment**
    - Create a `conda` Python environment and activate it:
        ```sh
        conda create --name virtualenv python==3.8.0
        conda activate virtualenv
        ```
    - lone this repository recursively to have pybind11
        ```sh
        git clone --recursive https://github.com/linh-gist/AdaptConfByteTrack.git
        ```
    - Install Packages
        ```sh
        numpy==1.23.1
        opencv-python==4.9.0.80
        loguru==0.7.2
        scipy==1.10.1
        lap==0.5.12
        cython_bbox==0.1.5
        matplotlib==3.5.3
        filterpy==1.4.5
        motmetrics==1.4.0
        openpyxl==3.1.5
        pycocotools==2.0.7
        tabulate==0.9.0
        # git clone https://github.com/JonathonLuiten/TrackEval.git
        # cd TrackEval, python setup.py build develop
        ```

2. **Prepare Data**
    - Datasets: 
        - MOT16, MOT17, MOT20
        - You can also run with your custom dataset but need a detector

3. **Run the Tracking Demo**
   - Change parameters in `make_parser()` in `track.py` such as `use_gmc`, `data_dir` (MOTChallenge GT data)
   - Run `python track.py`


### Citation
If you find this project useful in your research, please consider citing by:

```
@inproceedings{van2023adaptive,
  title={Adaptive Confidence Threshold for ByteTrack in Multi-Object Tracking},
  author={Linh Van Ma and Muhammad Ishfaq Hussain and JongHyun Park and Jeongbae Kim and Moongu Jeon},
  booktitle={2023 12th International Conference on Control, Automation and Information Sciences (ICCAIS)},
  pages={370--374},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgement
A part of the code is borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack), [OC-SORT](https://github.com/noahcao/OC_SORT), [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [FairMOT](https://github.com/ifzhang/FairMOT) and [SORT](https://github.com/abewley/sort). Thanks for their wonderful works.
