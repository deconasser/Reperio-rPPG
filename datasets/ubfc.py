from datasets.base import BaseDataset
import numpy as np
import pandas as pd
import glob
import os
import cv2
from tqdm import tqdm

class UBFCDataset(BaseDataset):
    def preprocess(self):
        data_dirs = sorted(glob.glob(os.path.join(self.data_root, 'subject*')))
        subjects = sorted([int(os.path.split(d)[-1].removeprefix("subject")) for d in data_dirs])
        
        train_subjects = subjects[:int(0.72 * len(subjects))]
        test_subjects = subjects[int(0.72 * len(subjects)):]
        
        assert set(train_subjects).intersection(test_subjects) == set()
        
        fold1 = subjects[:int(0.72 * len(subjects))]
        fold2 = subjects[int(0.72 * len(subjects)):]
        splits = pd.DataFrame(columns=['filename', 'split', 'fold'])
        
        for dir in tqdm(data_dirs, desc='Preprocessing'):
            subject = int(os.path.split(dir)[-1].removeprefix("subject"))
            if subject in train_subjects:
                split = 'train'
            elif subject in test_subjects:
                split = 'test'
            else:
                raise ValueError('Subject not in any split')
                
            if subject in fold1:
                fold = 1
            elif subject in fold2:
                fold = 0
            else:
                raise ValueError('Subject not in any fold')
            
            vid_file = os.path.join(dir, 'vid.avi')
            gt_file = os.path.join(dir, 'ground_truth.txt')
            frames = self.read_video(vid_file)
            waves = self.read_wave(gt_file)
            
            splits = splits.append(self.save(frames, waves, subject, 0, split, fold), ignore_index=True)
        
        splits.to_csv(os.path.join('datasets', 'ubfc_splits.csv'), index=False)

    @staticmethod
    def read_video(video_file):
        """Read video file, return an array of (T, H, W, 3)."""
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = []
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = VidObj.read()
        frames = np.asarray(frames)
        VidObj.release()
        return frames

    @staticmethod
    def read_wave(bvp_file):
        """Read BVP signal (ground truth) and return numpy array."""
        with open(bvp_file, "r") as f:
            content = f.read()
            first_line = content.split("\n")[0]
            bvp = [float(x) for x in first_line.split()]
        return np.asarray(bvp)
