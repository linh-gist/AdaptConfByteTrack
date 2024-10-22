from loguru import logger

from trackers.bytetrack.byte_tracker import BYTETracker
from trackers.sort.sort import Sort
# from trackers.ocsort_tracker.ocsort import OCSort
from trackers.ocsort.ocsort import OCSort
from trackers.fairmot.multitracker import JDETracker

import os
import numpy as np
import cv2
import glob
import os.path as osp
import time
from tracking_utils.evaluation import Evaluator
import motmetrics as mm
import trackeval


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def mot16(root):
    seqs_train = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
    seqs_test = ['MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14']
    train_dir = root + '/MOT16/train'
    test_dir = root + '/MOT16/test'
    return train_dir, test_dir, seqs_train, seqs_test


def mot17(root):
    seqs_train = ['MOT17-02-DPM', 'MOT17-02-FRCNN', 'MOT17-02-SDP', 'MOT17-04-DPM', 'MOT17-04-FRCNN', 'MOT17-04-SDP',
                  'MOT17-05-DPM', 'MOT17-05-FRCNN', 'MOT17-05-SDP', 'MOT17-09-DPM', 'MOT17-09-FRCNN', 'MOT17-09-SDP',
                  'MOT17-10-DPM', 'MOT17-10-FRCNN', 'MOT17-10-SDP', 'MOT17-11-DPM', 'MOT17-11-FRCNN', 'MOT17-11-SDP',
                  'MOT17-13-DPM', 'MOT17-13-FRCNN', 'MOT17-13-SDP']
    seqs_test = ['MOT17-01-DPM', 'MOT17-01-FRCNN', 'MOT17-01-SDP', 'MOT17-03-DPM', 'MOT17-03-FRCNN', 'MOT17-03-SDP',
                 'MOT17-06-DPM', 'MOT17-06-FRCNN', 'MOT17-06-SDP', 'MOT17-07-DPM', 'MOT17-07-FRCNN', 'MOT17-07-SDP',
                 'MOT17-08-DPM', 'MOT17-08-FRCNN', 'MOT17-08-SDP', 'MOT17-12-DPM', 'MOT17-12-FRCNN', 'MOT17-12-SDP',
                 'MOT17-14-DPM', 'MOT17-14-FRCNN', 'MOT17-14-SDP']
    train_dir = root + '/MOT17/train'
    test_dir = root + '/MOT17/test'
    return train_dir, test_dir, seqs_train, seqs_test


def mot20(root):
    seqs_train = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
    seqs_test = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08']
    train_dir = root + '/MOT20/train'
    test_dir = root + '/MOT20/test'
    return train_dir, test_dir, seqs_train, seqs_test


def mot17_eval(result_dir="./results/bytetrack"):
    data_root = 'D:/dataset/tracking/mot'
    train_dir, test_dir, seqs_train, seqs_test = mot17(data_root)
    seqs = ["02", "04", "05", "09", "10", "11", "13"]  # , "01", "03", "06", "07", "08", "12", "14"]
    sub_seqs = ["DPM", "SDP", "FRCNN"]
    for seq in seqs:
        input_file = result_dir + "/MOT16-" + seq + ".txt"
        for sub_seq in sub_seqs:
            output_file = result_dir + "/MOT17-" + seq + "-" + sub_seq + ".txt"
            with open(input_file) as f:
                with open(output_file, "w") as f1:
                    for line in f:
                        f1.write(line)
    accs = []
    for seq in seqs_train:
        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(train_dir, seq, 'mot')
        accs.append(evaluator.eval_file(os.path.join(result_dir, seq + '.txt')))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs_train, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_dir, 'summary_{}.xlsx'.format(seqs_train[0].split('-')[0])))
    with open(os.path.join(result_dir, 'summary_{}.txt'.format(seqs_train[0].split('-')[0])), 'w') as f:
        f.write(strsummary)


def evaluate_results(train_dir, result_dir, seqs_train):
    accs = []
    for seq in seqs_train:
        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(train_dir, seq, 'mot')
        accs.append(evaluator.eval_file(os.path.join(result_dir, seq + '.txt')))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs_train, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_dir, 'summary_{}.xlsx'.format(seqs_train[0].split('-')[0])))
    with open(os.path.join(result_dir, 'summary_{}.txt'.format(seqs_train[0].split('-')[0])), 'w') as f:
        f.write(strsummary)


def evaluate_hota_trackeval(seqs, gt_folder='./data/MOT16-train', trackers_folder='./results/'):
    output_folder = './results/'

    # List of sequences to evaluate
    sequences = {seq: None for seq in seqs}
    # sequences = {
    #     'MOT16-02': None,
    #     'MOT16-04': None,
    #     'MOT16-05': None,
    #     'MOT16-09': None,
    #     'MOT16-10': None,
    #     'MOT16-11': None,
    #     'MOT16-13': None
    # }

    # Configuration for the evaluation
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'TRACKERS_TO_EVAL': [''],  # List of trackers to evaluate
        'DATASETS_TO_EVAL': ['mot_challenge'],  # List of datasets to evaluate
        'BENCHMARK': 'MOT16',  # The benchmark to use (e.g., 'MOT17', 'MOT20')
        # 'SPLIT_TO_EVAL': 'train',  # Which split to evaluate ('train', 'test')
        'SKIP_SPLIT_FOL': True,
        'METRICS': ['HOTA'],  # Metrics to evaluate
        'OUTPUT_FOLDER': output_folder,
        'TRACKERS_FOLDER': trackers_folder,
        'GT_FOLDER': gt_folder,
        'PRINT_CONFIG': False,  # Disable printing of the configuration
        'TRACKER_SUB_FOLDER': '',
        'OUTPUT_SUB_FOLDER': '',
        'SEQ_INFO': sequences
    }

    # Create an Evaluator
    evaluator = trackeval.Evaluator(eval_config)

    # Load datasets
    dataset_list = []
    for dataset in eval_config['DATASETS_TO_EVAL']:
        if dataset == 'mot_challenge':
            dataset_list.append(trackeval.datasets.MotChallenge2DBox(eval_config))
        else:
            raise ValueError(f"Dataset {dataset} is not supported in this example.")

    # Load metrics
    metrics_list = []
    for metric in eval_config['METRICS']:
        if metric == 'HOTA':
            metrics_list.append(trackeval.metrics.HOTA(eval_config))
        else:
            raise ValueError(f"Metric {metric} is not supported in this example.")

    # Run evaluation
    output_res, _ = evaluator.evaluate(dataset_list, metrics_list)

    # Print the results
    # for metric_res in output_res:
    #     print(metric_res)


class MOTEvaluator:
    def __init__(self, args):
        self.args = args
        self.show_image = args.show_image

    def evaluate_BYTETrack(self, dets_path):
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh

        train_dir, test_dir, seqs_train, seqs_test = mot16(self.args.data_dir)
        for video_name in seqs_train:
            print("Starting tracking sequence", video_name)
            results = []
            dets = np.load(dets_path + "/" + video_name + ".npy")
            n_frames = int(np.amax(dets[:, 0]))
            img_path = os.path.join(train_dir, video_name)
            files = sorted(glob.glob(osp.join(img_path, 'img1') + '/*.jpg'))
            total_time = 0
            for frame_id in range(n_frames):
                if video_name == 'MOT16-05' or video_name == 'MOT16-06':
                    self.args.track_buffer = 14
                elif video_name == 'MOT16-13' or video_name == 'MOT16-14':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT16-01':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT16-06':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT16-12':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT16-14':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh
                if frame_id == 0:
                    tracker = BYTETracker(self.args)
                img0 = cv2.imread(files[frame_id])

                frame_bboxes = dets[dets[:, 0] == frame_id, :][:, 1:]
                # run tracking
                start = time.time()
                online_targets = tracker.update(frame_bboxes, img0)
                total_time += time.time() - start
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tid, tlwh = int(t[0]), t[1:]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        # online_scores.append(t.score)
                        if self.show_image:
                            l, t = int(tlwh[0]), int(tlwh[1])
                            r, b = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
                            cxy = (int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3] / 2))
                            # draw bbox
                            img0 = cv2.circle(img0, cxy, radius=8, color=(255, 255, 255), thickness=-1)
                            img0 = cv2.rectangle(img0, (l, t), (r, b), color=(255, 255, 255), thickness=2)
                            img0 = cv2.putText(img0, str(tid), org=cxy, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=0.65,
                                               color=(0, 255, 255), thickness=2)
                # save results
                results.append((frame_id + 1, online_tlwhs, online_ids))
                if self.show_image:
                    str_show = 'Adaptive Conf. (Frame {}'.format(frame_id) + ")"
                    img0 = cv2.putText(img0, str_show, org=(img0.shape[1] - 450, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=1, color=(255, 0, 255), thickness=2)
                    scale_percent = 0.6  # percent of original size
                    dim = (int(img0.shape[1] * scale_percent), int(img0.shape[0] * scale_percent))
                    resized = cv2.resize(img0, dim, interpolation=cv2.INTER_AREA)  # resize image
                    cv2.imshow('Image', resized)
                    cv2.moveWindow('Image', 200, 200)
                    cv2.waitKey(1)
                    cv2.imwrite("./results/images/" + str(frame_id) + ".jpg", img0)
                if frame_id == n_frames - 1:
                    result_filename = os.path.join(self.args.result_dir, '{}.txt'.format(video_name))
                    write_results_no_score(result_filename, results)
            print("BYTETrack FPS: " + video_name, round(1.0 / (total_time / n_frames), 2))
        evaluate_results(train_dir, self.args.result_dir, seqs_train)
        evaluate_hota_trackeval(seqs_train, train_dir, self.args.result_dir)

    def evaluate_sort(self, dets_path):
        tracker = Sort(self.args.track_thresh)
        train_dir, test_dir, seqs_train, seqs_test = mot16(self.args.data_dir)
        for video_name in seqs_train:
            print("Starting tracking sequence", video_name)
            results = []
            dets = np.load(dets_path + "/" + video_name + ".npy")
            n_frames = int(np.amax(dets[:, 0]))
            if self.show_image:
                img_path = os.path.join(train_dir, video_name)
                files = sorted(glob.glob(osp.join(img_path, 'img1') + '/*.jpg'))
            total_time = 0
            for frame_id in range(n_frames):
                if self.show_image:
                    img0 = cv2.imread(files[frame_id])
                frame_bboxes = dets[dets[:, 0] == frame_id, :][:, 1:]
                # run tracking
                start = time.time()
                online_targets = tracker.update(frame_bboxes)
                total_time += time.time() - start
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = int(t[4])
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        if self.show_image:
                            l, t = int(tlwh[0]), int(tlwh[1])
                            r, b = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
                            cxy = (int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3] / 2))
                            # draw bbox
                            img0 = cv2.circle(img0, cxy, radius=8, color=(255, 255, 255), thickness=-1)
                            img0 = cv2.rectangle(img0, (l, t), (r, b), color=(255, 255, 255), thickness=2)
                            img0 = cv2.putText(img0, str(tid), org=cxy, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=0.65,
                                               color=(0, 255, 255), thickness=2)
                # save results
                results.append((frame_id + 1, online_tlwhs, online_ids))
                if self.show_image:
                    str_show = 'Frame {}'.format(frame_id)
                    img0 = cv2.putText(img0, str_show, org=(img0.shape[1] - 400, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=1, color=(255, 255, 255), thickness=2)
                    scale_percent = 0.6  # percent of original size
                    dim = (int(img0.shape[1] * scale_percent), int(img0.shape[0] * scale_percent))
                    resized = cv2.resize(img0, dim, interpolation=cv2.INTER_AREA)  # resize image
                    cv2.imshow('Image', resized)
                    cv2.moveWindow('Image', 200, 200)
                    cv2.waitKey(1)
                if frame_id == n_frames - 1:
                    result_filename = os.path.join(self.args.result_dir, '{}.txt'.format(video_name))
                    write_results_no_score(result_filename, results)
            print("SORT FPS: " + video_name, round(1.0 / (total_time / n_frames), 2))
        evaluate_results(train_dir, self.args.result_dir, seqs_train)
        evaluate_hota_trackeval(seqs_train, train_dir, self.args.result_dir)

    def evaluate_ocsort(self, args, dets_path):
        train_dir, test_dir, seqs_train, seqs_test = mot16(self.args.data_dir)
        for video_name in seqs_train:
            tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, asso_func=args.asso,
                             delta_t=args.deltat, inertia=args.inertia, use_byte=args.use_byte, use_gmc=args.use_gmc)
            print("Starting tracking sequence", video_name)
            results = []
            dets = np.load(dets_path + "/" + video_name + ".npy")
            n_frames = int(np.amax(dets[:, 0]))
            # if self.show_image:
            img_path = os.path.join(train_dir, video_name)
            files = sorted(glob.glob(osp.join(img_path, 'img1') + '/*.jpg'))
            total_time = 0
            for frame_id in range(n_frames):
                # if self.show_image:
                img0 = cv2.imread(files[frame_id])
                frame_bboxes = dets[dets[:, 0] == frame_id, :][:, 1:]
                # run tracking
                start = time.time()
                online_targets = tracker.update(frame_bboxes, img0)
                total_time += time.time() - start
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[1], t[2], t[3], t[4]]
                    tid = int(t[0])
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        if self.show_image:
                            l, t = int(tlwh[0]), int(tlwh[1])
                            r, b = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
                            cxy = (int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3] / 2))
                            # draw bbox
                            img0 = cv2.circle(img0, cxy, radius=8, color=(255, 255, 255), thickness=-1)
                            img0 = cv2.rectangle(img0, (l, t), (r, b), color=(255, 255, 255), thickness=2)
                            img0 = cv2.putText(img0, str(tid), org=cxy, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=0.65,
                                               color=(0, 255, 255), thickness=2)
                # save results
                results.append((frame_id + 1, online_tlwhs, online_ids))
                if self.show_image:
                    str_show = 'Frame {}'.format(frame_id)
                    img0 = cv2.putText(img0, str_show, org=(img0.shape[1] - 400, 30),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=1, color=(255, 255, 255), thickness=2)
                    scale_percent = 0.6  # percent of original size
                    dim = (int(img0.shape[1] * scale_percent), int(img0.shape[0] * scale_percent))
                    resized = cv2.resize(img0, dim, interpolation=cv2.INTER_AREA)  # resize image
                    cv2.imshow('Image', resized)
                    cv2.moveWindow('Image', 200, 200)
                    cv2.waitKey(1)
                if frame_id == n_frames - 1:
                    result_filename = os.path.join(self.args.result_dir, '{}.txt'.format(video_name))
                    write_results_no_score(result_filename, results)
            print("OCSORT FPS: " + video_name, round(1.0 / (total_time / n_frames), 2))
        evaluate_results(train_dir, self.args.result_dir, seqs_train)
        evaluate_hota_trackeval(seqs_train, train_dir, self.args.result_dir)

    def evaluate_fairmot(self, dets_path):
        tracker = JDETracker(self.args)
        train_dir, test_dir, seqs_train, seqs_test = mot16(self.args.data_dir)
        for video_name in seqs_train:
            print("Starting tracking sequence", video_name)
            results = []
            dets = np.load(dets_path + "/" + video_name + ".npz")
            n_frames = int(len(dets.files) / 2)
            if self.show_image:
                img_path = os.path.join(train_dir, video_name)
                files = sorted(glob.glob(osp.join(img_path, 'img1') + '/*.jpg'))
            for frame_id in range(n_frames):
                if self.show_image:
                    img0 = cv2.imread(files[frame_id])
                # run tracking
                online_targets = tracker.update(dets, frame_id)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = int(t.track_id)
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        if self.show_image:
                            l, t = int(tlwh[0]), int(tlwh[1])
                            r, b = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
                            cxy = (int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3] / 2))
                            # draw bbox
                            img0 = cv2.circle(img0, cxy, radius=8, color=(255, 255, 255), thickness=-1)
                            img0 = cv2.rectangle(img0, (l, t), (r, b), color=(255, 255, 255), thickness=2)
                            img0 = cv2.putText(img0, str(tid), org=cxy, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=0.65,
                                               color=(0, 255, 255), thickness=2)
                # save results
                results.append((frame_id, online_tlwhs, online_ids))
                if self.show_image:
                    str_show = 'Frame {}'.format(frame_id)
                    img0 = cv2.putText(img0, str_show, org=(img0.shape[1] - 400, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=1, color=(255, 255, 255), thickness=2)
                    scale_percent = 0.6  # percent of original size
                    dim = (int(img0.shape[1] * scale_percent), int(img0.shape[0] * scale_percent))
                    resized = cv2.resize(img0, dim, interpolation=cv2.INTER_AREA)  # resize image
                    cv2.imshow('Image', resized)
                    cv2.moveWindow('Image', 200, 200)
                    cv2.waitKey(1)
                if frame_id == n_frames - 1:
                    result_filename = os.path.join(self.args.result_dir, '{}.txt'.format(video_name))
                    write_results_no_score(result_filename, results)
        evaluate_results(train_dir, self.args.result_dir, seqs_train)


def get_color(idx):
    idx = (idx + 1) * 50
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def vdemo_from_rfile(rfile, img_folder):
    tracks = np.loadtxt(rfile, delimiter=",")
    # '{frame},{id},{top},{left},{w},{h},-1,-1,-1,-1\n'
    files = sorted(glob.glob(img_folder + '/*.jpg'))
    n_frames = len(files)
    video_name = os.path.join(os.path.dirname(rfile), os.path.basename(rfile).split(".")[0] + ".mp4")
    size = cv2.imread(files[0]).shape[0:2][::-1]
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for frame_id in range(n_frames):
        img0 = cv2.imread(files[frame_id])
        frame_tracks = tracks[tracks[:, 0] == frame_id + 1]
        for tt in frame_tracks:
            tlwh = tt[2:6]
            tid = int(tt[1])
            l, t = int(tlwh[0]), int(tlwh[1])
            r, b = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
            # cxy = (int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3] / 2))
            # draw bbox
            color = get_color(tid)
            # img0 = cv2.circle(img0, cxy, radius=8, color=color, thickness=-1)
            img0 = cv2.rectangle(img0, (l, t), (r, b), color=color, thickness=2)
            img0 = cv2.putText(img0, str(tid), org=(l, t + 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65,
                               color=color, thickness=2)
        cv2.imshow(rfile, img0)
        cv2.waitKey(1)
        out.write(img0)
    out.release()


if __name__ == "__main__":
    # mot17_eval()
    seq = "MOT16-02"
    vdemo_from_rfile(f"results/bytetrack/bytetrack/{seq}.txt",
                     f"/media/ubuntu/2715608D71CBF6FC/datasets/mot/MOT16/train/{seq}/img1/")
