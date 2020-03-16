import re, os, time
import cv2, json, random 

import numpy as np
import config.config as dfc

class DataSplitter:
#{
    def __init__(self, ipart=0, validation_split=0.2, data_subset=1.0, shuffle=True):
    #{
        self.valid_split = []
        self.train_split = []            

        videonames = list(filter(re.compile(r'\w+\.mp4').match, os.listdir(traindir(ipart))))
        
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

# Given a video name pair: vname a deepfake, and oname its original/parent, function creates
# "fakerframe" images (scaled-difference-blend-mode-images), cropped to size crop, starting
# from video frame: fentry and then contiguously for: nframes. Also creates a 'fakerprint' 
# per pair, which is just the scaled sum the fakerframes. All images files saved to datapath.
def create_diff_frames(vname, oname, datapath, fentry=120, nframes=60, crop=(720,1280)):
#{
    video = cv2.VideoCapture(vname)
    orig = cv2.VideoCapture(oname)
    
    video.set(cv2.CAP_PROP_POS_FRAMES, fentry)
    orig.set(cv2.CAP_PROP_POS_FRAMES, fentry)

    # Get orientation from video header
    fwidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_portrait_orient, lft, rht, bot = _crop_params(
        fwidth, fheight, crop)
    
    zblock = np.zeros((crop[0], crop[1], 3), dtype=np.uint8)
    fidx, vsuccess, osuccess, fakerprint = 0, True, True, None
    while video.isOpened() and orig.isOpened() and vsuccess and osuccess and fidx < nframes:
    #{
        # Fakerframes are scaled Photoshop difference blend-mode images,
        # see: https://helpx.adobe.com/photoshop/using/blending-modes.htm

        vsuccess, videoframe = video.read()
        osuccess, origframe = orig.read()
        
        if vsuccess and osuccess:
        #{
            # Crop 
            videoframe = videoframe[0:bot,lft:rht,:]
            origframe = origframe[0:bot,lft:rht,:]

            # Rotate to landscape
            if is_portrait_orient:
                videoframe = cv2.rotate(videoframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
                origframe = cv2.rotate(origframe, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Diff and scale (usually up to fill 255 range) 
            fakerframe = cv2.subtract(cv2.max(videoframe, origframe), cv2.min(videoframe, origframe))
            fakerframe = cv2.scaleAdd(fakerframe, 255/fakerframe.max(), zblock) # faster than np.mult
            cv2.imwrite(f"{datapath}/fakerframe{fentry+fidx}.jpg", fakerframe)
            
            # Effectively fakerprint = reduce(sum, fakerframes)
            if fakerprint is None: fakerprint = fakerframe.astype(np.uint16)
            else: fakerprint = fakerprint + fakerframe # faster than cv2.add
        #}

        fidx += 1
    #}
    
    # Scale (usually down to fill 255 range)
    fakerprint = cv2.scaleAdd(fakerprint, 255/fakerprint.max(), zblock.astype(np.uint16))
    cv2.imwrite(f"{datapath}/fakerprint.jpg", fakerprint)
    
    video.release()
    orig.release()
#}

def _crop_params(fwidth, fheight, crop):
#{
    is_portrait_orient = fwidth < fheight
    lft = (fwidth - crop[0])//2 if is_portrait_orient else (fwidth - crop[1])//2
    rht = lft + crop[0] if is_portrait_orient else lft + crop[1]
    bot = crop[1] if is_portrait_orient else crop[0]
    return is_portrait_orient, lft, rht, bot
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

def _load_video(videoname, nsamps=30, fentry=120, nframes=60, crop=(720,1280)):
#{
    video = cv2.VideoCapture(f"{traindir()}/{videoname}")
    video.set(cv2.CAP_PROP_POS_FRAMES, fentry)

    # Get orientation from video header
    fwidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_portrait_orient, lft, rht, bot = _crop_params(
        fwidth, fheight, crop)

    fidx = 0
    while video.isOpened() and fidx < nframes:
    #{
        samples = np.empty((nsamps, crop[0], crop[1], 3), dtype=np.uint8)
        for i in range(nsamps):
        #{
            vsuccess, videoframe = video.read()
            assert vsuccess, f"{videoname} frame {fentry+fidx+i} read failed"

            # Crop (and rotate if portrait orient)
            videoframe = videoframe[0:bot,lft:rht,:]
            if is_portrait_orient: videoframe = cv2.rotate(
                videoframe, cv2.ROTATE_90_COUNTERCLOCKWISE)

            samples[i,:,:,:] = videoframe
        #}

        yield samples
        fidx += nsamps
    #}

    video.release()
#}

def _load_target(videoname, nsamps=30, fentry=120, nframes=60):
#{
    fidx, fkdir = 0, f"{fakerdir()}/{videoname[:-4]}"
    
    while fidx < nframes:
    #{
        targets = np.zeros((nsamps, ), dtype=float)
        if os.path.isdir(fkdir):
        #{
            for i in range(nsamps):
                fakerframe = cv2.imread(f"{fkdir}/fakerframe{fentry+fidx+i}.jpg")                
                targets[i] = cv2.mean(fakerframe[:,:,1])[0] # Just use G-channel
        #}

        yield targets
        fidx += nsamps
    #}
#}

# def _load_video(videoname, nsamps=30):
# #{
#     video = cv2.VideoCapture(f"{traindir()}/{videoname}")

#     # Get orientation from video header
#     fwidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     is_portrait_orient = fwidth < fheight

#     fidx = 0
#     while video.isOpened() and fidx < 300:
#     #{
#         samples = np.empty((nsamps, 1080, 1920, 3), dtype=np.uint8)
#         for i in range(nsamps):
#         #{
#             vsuccess, videoframe = video.read()
#             assert vsuccess, f"{videoname} frame {fidx+i} read failed"
#             if is_portrait_orient: videoframe = cv2.rotate(
#                 videoframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
#             samples[i,:,:,:] = videoframe
#         #}

#         yield samples
#         fidx += nsamps
#     #}

#     video.release()
# #}

# def _load_target(videoname, nsamps=30):
# #{
#     fidx, fkdir = 0, f"{fakerdir()}/{videoname[:-4]}"
    
#     while fidx < 300:
#     #{
#         targets = np.zeros((nsamps, ), dtype=float)
#         if os.path.isdir(fkdir):
#         #{
#             for i in range(nsamps):
#                 fakerframe = cv2.imread(f"{fkdir}/fakerframe{fidx+i}.jpg")
#                 targets[i] = cv2.mean(fakerframe[:,:,1])[0] # Just use G-channel
#         #}

#         yield targets
#         fidx += nsamps
#     #}
# #}
