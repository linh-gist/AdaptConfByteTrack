import argparse
from mot_evaluator import MOTEvaluator


def make_parser():
    parser = argparse.ArgumentParser("Object State Eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument("--data_dir", default="/media/ubuntu/2715608D71CBF6FC/datasets/mot", type=str, help="eval seed")
    parser.add_argument("--result_dir", default="./results/bytetrack", type=str, help="eval seed")
    parser.add_argument("--show_image", default=False, type=bool, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")
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
    # parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
    # add cmc for fixing the camera motion
    parser.add_argument("--use_gmc", default=True, type=bool, help="use Camera Motion Compensation")

    return parser


if __name__ == "__main__":
    # import cv2
    # import glob
    # import os.path as osp
    # import numpy as np
    # fixedfiles = sorted(glob.glob('./results/images_fixed/*.jpg'))
    # adaptfiles = sorted(glob.glob('./results/images/*.jpg'))
    # for ii, (adapt, fixed) in enumerate(zip(adaptfiles, fixedfiles)):
    #     im_adapt = cv2.imread(adapt)
    #     im_fixed = cv2.imread(fixed)
    #     concat = np.concatenate((im_adapt, im_fixed), axis=1)
    #     cv2.imwrite("./results/images_results/" + str(ii)+".jpg", concat)

    args = make_parser().parse_args()
    evaluator = MOTEvaluator(args)
    for det in ["bytetrack"]:
        #
        args.result_dir = "./results/{}/bytetrack".format(det)
        evaluator.evaluate_BYTETrack("./detection/{}/".format(det))
        #
        # args.result_dir = "./results/{}/sort".format(det)
        # evaluator.evaluate_sort("./detection/{}/".format(det))
        #
        args.result_dir = "./results/{}/ocsort".format(det)
        evaluator.evaluate_ocsort(args, "./detection/{}/".format(det))

    # evaluator.evaluate_sort("./detection/bytetrack")
    # evaluator.evaluate_fairmot("./detection/detector_cstrack")
