import re, os
import cv2, json, random 

import numpy as np
import deepfake.config as dfc

class DataSplitter:
#{
    def __init__(self, ipart=0, validation_split=0.2, data_subset=1.0):
    #{
        self.valid_split = []
        self.train_split = []            

        videonames = list(filter(re.compile(r'\w+\.mp4').match, os.listdir(traindir())))
        
        np.random.shuffle(videonames)
        nvideos = int(len(videonames)*data_subset)
        vindices = set(random.sample(range(nvideos), int(validation_split*nvideos)))

        for i in range(nvideos):
            if i in vindices: self.valid_split.append(videonames[i])
            else: self.train_split.append(videonames[i])
    #}
#}

def file_exists(filename):
    try:
        with open(filename) as video: return True
    except FileNotFoundError: return False

def trainpart(i=0): return dfc.DATA_TRAIN_PART.replace('IDX', str(i))
def traindir(i=0): return dfc.ROOT_DATA_TRAIN.replace('IDX', str(i))
def fakerdir(i=0): return dfc.ROOT_FAKER_FRAMES.replace('IDX', str(i))

# This fcn creates an array of the valid deepfake-original video name pairs
# based on the metadata.json file and the existence (or not) of the videos
def faked_video_pairs(ipart):
#{
    pc_pairs = []
    
    try:
    #{
        with open(f"{traindir(ipart)}/metadata.json") as jsonfile:
        #{
            metadata = json.load(jsonfile)
            for vidname, meta in metadata.items():
                vname = f"{traindir(ipart)}/{vidname}"
                if file_exists(vname) and (meta['label'] == 'FAKE'):
                    oname = f"{traindir(ipart)}/{meta['original']}"
                    if file_exists(oname): pc_pairs.append((vname, oname))
        #}
    #}  
    except PermissionError as err: print("ERROR:", err)
    return pc_pairs
#}

def lazy_load_partition(splitter, validation=False):
#{
    # Fcn loads only 30 frames per yield/batch. This is necessary to pass
    # Tensoflow assert that tensor size (~1080*1920*30*32) < sizeof(int32)
    videonames = splitter.valid_split if validation else splitter.train_split

    vidx = 0
    while True:
    #{
        xloader = _load_video(videonames[vidx])
        yloader = _load_target(videonames[vidx])
        for x_train, y_train in zip(xloader, yloader):
            yield x_train, y_train

        vidx += 1; vidx %= len(videonames)
    #}
#}

def _load_video(videoname, nsamps=30):
#{
    video = cv2.VideoCapture(f"{traindir()}/{videoname}")

    # Get orientation from video header
    fwidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_portrait_orient = fwidth < fheight

    fidx = 0
    while video.isOpened() and fidx < 300:
    #{
        samples = np.empty((nsamps, 1080, 1920, 3), dtype=np.uint8)
        for i in range(nsamps):
        #{
            vsuccess, videoframe = video.read()
            assert vsuccess, f"{videoname} frame {fidx+i} read failed"
            if is_portrait_orient: videoframe = cv2.rotate(
                videoframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
            samples[i,:,:,:] = videoframe
        #}

        yield samples
        fidx += nsamps
    #}

    video.release()
#}

def _load_target(videoname, nsamps=30):
#{
    fidx, fkdir = 0, f"{fakerdir()}/{videoname[:-4]}"
    
    while fidx < 300:
    #{
        targets = np.zeros((nsamps, ), dtype=float)
        if os.path.isdir(fkdir):
        #{
            for i in range(nsamps):
                fakerframe = cv2.imread(f"{fkdir}/fakerframe{fidx+i}.jpg")
                targets[i] = cv2.mean(fakerframe[:,:,1])[0] # Just use G-channel
        #}

        yield targets
        fidx += nsamps
    #}
#}
