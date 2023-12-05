from loguru import logger

from trackers.bytetrack.byte_tracker import BYTETracker
from trackers.sort.sort import Sort
from trackers.fairmot.multitracker import JDETracker

import os
import numpy as np
import cv2
import glob
import os.path as osp

from tracking_utils.evaluation import Evaluator
import motmetrics as mm


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


class MOTEvaluator:
    def __init__(self, args):
        self.args = args
        self.show_image = args.show_image

    def evaluate_BYTETrack(self, dets_path):
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh

        train_dir, test_dir, seqs_train, seqs_test = mot20(self.args.data_dir)
        for video_name in seqs_train:
            print("Starting tracking sequence", video_name)
            results = []
            dets = np.load(dets_path + "/" + video_name + ".npy")
            n_frames = int(np.amax(dets[:, 0]))
            if self.show_image:
                img_path = os.path.join(train_dir, video_name)
                files = sorted(glob.glob(osp.join(img_path, 'img1') + '/*.jpg'))
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
                if self.show_image:
                    img0 = cv2.imread(files[frame_id])

                frame_bboxes = dets[dets[:, 0] == frame_id, :][:, 1:]
                # run tracking
                online_targets = tracker.update(frame_bboxes)
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
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
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
                    write_results(result_filename, results)
            exit(1)
        evaluate_results(train_dir, self.args.result_dir, seqs_train)

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
            for frame_id in range(n_frames):
                if self.show_image:
                    img0 = cv2.imread(files[frame_id])
                frame_bboxes = dets[dets[:, 0] == frame_id, :][:, 1:]
                # run tracking
                online_targets = tracker.update(frame_bboxes)
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


if __name__ == "__main__":
    mot17_eval()
