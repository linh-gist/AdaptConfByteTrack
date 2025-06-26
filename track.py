import argparse
from mot_evaluator import MOTEvaluator


def make_parser():
    parser = argparse.ArgumentParser("Object State Eval")
    parser.add_argument("--data_dir", default="/media/ubuntu/2715608D71CBF6FC/datasets/mot", type=str, help="eval seed")
    parser.add_argument("--result_dir", default="./results/bytetrack", type=str, help="eval seed")
    parser.add_argument("--show_image", default=False, type=bool, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.3, help="detection confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    # tracking args OCSORT
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
    parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
    parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
    parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
    parser.add_argument("--use_byte", type=bool, default=True, help="use BYTE association")
    # add cmc for fixing the camera motion
    parser.add_argument("--use_gmc", default=True, type=bool, help="use Camera Motion Compensation")

    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    evaluator = MOTEvaluator(args)
    for detector in ['detector_poi', 'detector_trades', 'detector_gsdt', 'detector_jde', 'detector_cstrack',
                     'detector_fairmot128', 'detector_bytetrack', 'detectors_yolov11']:
        for tracker in ['SORT', 'BYTETrack', 'OCSort', 'DeepSort', 'MOTDT', 'FairMOT', 'DeepOCSort', 'LMB']:
            print(f"#################### {detector} #################### {tracker} ####################")
            args.result_dir = f"./results/{detector}/{tracker}"
            evaluator.evaluate_trackers(args, f"./dets/{detector}", tracker)
    ###############
