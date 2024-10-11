# BYTETrack, OC-SORT, CMC (Camera Motion Compensation), HOTA Evaluation
Source code for the paper "Adaptive Confidence Threshold for ByteTrack in Multi-Object Tracking"

> [**Adaptive Confidence Threshold for ByteTrack in Multi-Object Tracking**](https://arxiv.org/abs/2312.01650),            
> Linh Van Ma, Muhammad Ishfaq Hussain, JongHyun Park, Jeongbae Kim, Moongu Jeon,
> ICCAIS 2023 (The 12th International Conference on Control, Automation and Information Sciences), November 27th to 29th, 2023 in Hanoi


## How to run?
    # Build cython_bbox
    
    python3 track.py
    
    # Detection files outputed from detectors
    # Change detector => evaluate_fairmot("./detection/detector_cstrack")
    # Change detector => evaluate_fairmot("./detection/bytetrack")


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
