import re, os, time
import cv2, json, random 

import numpy as np
import config.config as dfc

class DataSplitter:
#{
    @staticmethod
    def videonames(datapath):
        return sorted(filter(re.compile(r'\w+\.mp4').match, os.listdir(datapath)))

    def __init__(self, videonames=None, validation_split=0.2, data_subset=1.0, shuffle=True):
    #{
        self.train_split = []
        self.valid_split = []

        if videonames is None: videonames = DataSplitter.videonames(traindir())
        
        if shuffle: np.random.shuffle(videonames)
        nvideos = int(len(videonames)*data_subset)
        vindices = set(random.sample(range(nvideos), int(validation_split*nvideos)))

        for i in range(nvideos):
            if i in vindices: self.valid_split.append(videonames[i])
            else: self.train_split.append(videonames[i])
    #}
#}

def file_exists(filename):
    try:
        with open(filename): return True
    except FileNotFoundError: return False

def trainpart(i=0): return dfc.DATA_TRAIN_PART.replace('IDX', str(i))
def traindir(i=0): return dfc.ROOT_DATA_TRAIN.replace('IDX', str(i))
def fakerdir(i=0): return dfc.ROOT_FAKER_FRAMES.replace('IDX', str(i))
def frectdir(i=0): return dfc.ROOT_FRECT_FRAMES.replace('IDX', str(i))
def testdir(): return dfc.ROOT_DATA_TEST

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

def _crop_params(fwidth, fheight, crop):
#{
    is_portrait_orient = fwidth < fheight
    lft = (fwidth - crop[1])//2 if is_portrait_orient else (fwidth - crop[0])//2
    rht = lft + crop[1] if is_portrait_orient else lft + crop[0]
    bot = crop[0] if is_portrait_orient else crop[1]
    return is_portrait_orient, lft, rht, bot
#}

def lazy_load_testdata(videonames):
#{
    vidx = 0
    while True:
    #{
        initial = time.time()
        for x_test in _load_video(f"{testdir()}/{videonames[vidx]}"):
            yield x_test.astype(np.float32)

        print(f"Batch {vidx} processing time: {time.time()-initial:.3f} sec")
        vidx += 1; vidx %= len(videonames)
    #}
#}

def lazy_load_partition(splitter, validation=False):
#{
    # Fcn loads only 30 frames per yield/batch. This is necessary to pass
    # Tensoflow assert that tensor size (~1080*1920*30*32) < sizeof(int32)
    videonames = splitter.valid_split if validation else splitter.train_split

    vidx = 0
    while True:
    #{
        #initial = time.time()
        xloader = _load_video(f"{traindir()}/{videonames[vidx]}")
        yloader = _load_target(f"{fakerdir()}/{videonames[vidx][:-4]}")
        for x_train, y_train in zip(xloader, yloader): yield x_train, y_train

        #print(f"Batch {vidx} processing time: {time.time()-initial:.3f} sec")
        vidx += 1; vidx %= len(videonames)
    #}
#}

def _load_video(videoname, nsamps=30, fentry=120, nframes=60, crop=(1280,720), scsize=(640,360)):
#{
    video = cv2.VideoCapture(videoname)
    video.set(cv2.CAP_PROP_POS_FRAMES, fentry)

    # Get orientation from video header
    fwidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_portrait_orient, lft, rht, bot = _crop_params(fwidth, fheight, crop)
    scorientsize = (scsize[1], scsize[0]) if is_portrait_orient else scsize

    fidx = 0
    while video.isOpened() and fidx < nframes:
    #{
        samples = np.empty((nsamps, scsize[1], scsize[0], 3), dtype=np.uint8)
        for i in range(nsamps):
        #{
            vsuccess, videoframe = video.read()
            assert vsuccess, f"{videoname} frame {fentry+fidx+i} read failed"

            # Crop, then scale image size (and finally rotate if portrait orientation)
            videoframe = cv2.resize(videoframe[0:bot,lft:rht,:], scorientsize, interpolation=cv2.INTER_AREA)
            if is_portrait_orient: videoframe = cv2.rotate(videoframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
            samples[i,:,:,:] = videoframe
        #}

        yield samples
        fidx += nsamps
    #}

    video.release()
#}

def _load_target(vfakerdir, nsamps=30, fentry=120, nframes=60, targsize=(547,347)):
#{
    fidx, flatsz = 0, (targsize[1] * targsize[0] * 3)

    while fidx < nframes:
    #{
        targets = np.zeros((nsamps, flatsz), dtype=np.uint8)
        if os.path.isdir(vfakerdir):
        #{
            for i in range(nsamps):
                targets[i,:] = cv2.imread(f"{vfakerdir}/fakerframe{fentry+fidx+i}.jpg").flatten()
        #}

        yield targets
        fidx += nsamps
    #}
#}

def _load_mean_target(vfakerdir, nsamps=30, fentry=120, nframes=60):
#{
    fidx = 0
    while fidx < nframes:
    #{
        targets = np.zeros((nsamps, ), dtype=float)
        if os.path.isdir(vfakerdir):
        #{
            for i in range(nsamps):
                fakerframe = cv2.imread(f"{vfakerdir}/fakerframe{fentry+fidx+i}.jpg")                
                targets[i] = cv2.mean(fakerframe[:,:,1])[0] # Just use G-channel
        #}

        yield targets
        fidx += nsamps
    #}
#}

