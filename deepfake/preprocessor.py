import time, re, os, cv2
import numpy as np

import config.config as dfc
import deepfake.dfutillib as df
import deepfake.postgresdb as pgdb

class Preprocessor:
#{
    def __init__(self, minqueued=5, eventloop=10, logbatch=10):
        self.minqueued = minqueued
        self.eventloop = eventloop
        self.logbatch = logbatch
        self.maxblkid = self._queryblkid()

    def _queryblkid(self):
        sql = "SELECT MAX(blk_id) FROM videos"
        with pgdb.PostgreSqlHandle() as db_handle:
            return db_handle.sqlquery(sql, fetch='one')[0]

    def run(self):
    #{
        unpackfcn = lambda vt: pgdb.VideoTuple(*vt)
        countsql = ("SELECT COUNT(*) FROM epoch_queue WHERE "
                    "split = 'train' and status != 'COMPLETE'")

        while True:
        #{
            # Check for halt flag
            with pgdb.PostgreSqlHandle() as db_handle:
                if (db_handle.sqlquery(pgdb.EpochTuple.haltsql, fetch='one')[0] > 0):
                    break

            # Poll epoch_queue
            qcount = None
            with pgdb.PostgreSqlHandle() as db_handle:
                qcount = db_handle.sqlquery(countsql, fetch='one')[0]

            # Sleep or process and append to epoch_queue
            if self.minqueued is not None and qcount >= self.minqueued:
                time.sleep(self.eventloop)
            
            else: 
            #{
                # Select a random block from videos table
                blk_id = np.random.randint(self.maxblkid+1)
                blksql = (f"SELECT * FROM videos WHERE blk_id = {blk_id} " 
                            "AND proc_flg = FALSE AND label = 'FAKE' "
                            "AND (split = 'train' OR split = 'validate')")
                
                vidtuples = []
                with pgdb.PostgreSqlHandle() as db_handle:
                    dbresult = db_handle.sqlquery(blksql, fetch='all')
                    vidtuples = [vtup for vtup in map(unpackfcn, dbresult)]
                
                # If needed, preprocess before appending
                totproctime, proctimes = time.time(), []
                print(f"Preprocessing epoch block {blk_id}:")
                for i, vtup in enumerate(vidtuples):
                #{
                    # Create directories if needed
                    proctimes.append(time.time())
                    if not os.path.isdir(df.fakerdir(vtup.partition)): 
                        os.mkdir(df.fakerdir(vtup.partition))

                    vdir = re.split(r'[/.]', vtup.vidname)[-2]
                    datapath = f"{df.fakerdir(vtup.partition)}/{vdir}"
                    if not os.path.isdir(datapath): os.mkdir(datapath)
                    
                    # Create and save diff images
                    videourl = f"{df.traindir(vtup.partition)}/{vtup.vidname}"
                    origurl = f"{df.traindir(vtup.partition)}/{vtup.origname}"
                    self.create_diff_frames(videourl, origurl, datapath)
                    
                    # Logging
                    proctimes[-1] = time.time() - proctimes[-1]
                    if len(proctimes) == self.logbatch:
                        print(f"  Processed {i+1} videos, running average time/pair:",
                              f"{np.average(proctimes):.2f} sec")
                        proctimes = []
                #}
                print(f"  Total process time: {(time.time() - totproctime)/60.0:.2f} min")

                # Update preprocess flag on videos
                with pgdb.PostgreSqlHandle() as db_handle:
                    db_handle.sqlquery(f"UPDATE videos SET proc_flg = TRUE WHERE blk_id = {blk_id}")

                # Insert epoch block onto the queue
                with pgdb.PostgreSqlHandle() as db_handle:
                    etraintup = pgdb.EpochTuple(blk_id=blk_id, split='train', status='QUEUED')
                    evalidtup = pgdb.EpochTuple(blk_id=blk_id, split='validate', status='QUEUED')
                    db_handle.sqlquery(etraintup.insertsql()); db_handle.sqlquery(evalidtup.insertsql())
            #}
        #}
    #}
    
    # Given a video name pair: vname a deepfake, and oname its original/parent, function creates
    # "fakerframe" images (scaled-difference-blend-mode-images), cropped to size crop, starting
    # from video frame: fentry and then contiguously for: nframes. Also creates a 'fakerprint' 
    # per pair, which is just the scaled sum the fakerframes. All images files saved to datapath.
    def create_diff_frames(self, vname, oname, datapath, nframes=60, crop=(1280,720), targsize=dfc.TARGETSZ):
    #{
        video = cv2.VideoCapture(vname)
        orig = cv2.VideoCapture(oname)
        
        # Calc entry frame to center capture window
        fps = round(video.get(cv2.CAP_PROP_FPS))
        fentry = round((fps*10/2) - (nframes/2))
        tframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        assert fentry >= 0 and fentry+nframes <= tframes,\
            (f"Invalid frame window: [{fentry}, {fentry+nframes}] "
             f"At FPS: {fps}, only {tframes} frames in {vname}")

        video.set(cv2.CAP_PROP_POS_FRAMES, fentry)
        orig.set(cv2.CAP_PROP_POS_FRAMES, fentry)

        # Get orientation from video header
        fwidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        is_portrait_orient, lft, rht, bot = df.crop_params(fwidth, fheight, crop)
        targorientsize = (targsize[1], targsize[0]) if is_portrait_orient else targsize

        zblock = np.zeros((targsize[1], targsize[0], 3), dtype=np.uint8)
        fidx, vsuccess, osuccess, fakerprint = 0, True, True, None
        while video.isOpened() and orig.isOpened() and vsuccess and osuccess and fidx < nframes:
        #{
            # Fakerframes are scaled Photoshop difference blend-mode images,
            # see: https://helpx.adobe.com/photoshop/using/blending-modes.htm

            vsuccess, videoframe = video.read()
            osuccess, origframe = orig.read()
            
            if vsuccess and osuccess:
            #{
                # Crop and then scale image to the CONVNet target size
                videoframe = cv2.resize(videoframe[0:bot,lft:rht,:], targorientsize, interpolation=cv2.INTER_AREA)
                origframe = cv2.resize(origframe[0:bot,lft:rht,:], targorientsize, interpolation=cv2.INTER_AREA)

                # Rotate to landscape
                if is_portrait_orient:
                    videoframe = cv2.rotate(videoframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    origframe = cv2.rotate(origframe, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Diff and scale pixel values (usually scale up to fill [0,255] range) 
                fakerframe = cv2.subtract(cv2.max(videoframe, origframe), cv2.min(videoframe, origframe))
                fakerframe = cv2.scaleAdd(fakerframe, 255/fakerframe.max(), zblock) # faster than np.mult
                cv2.imwrite(f"{datapath}/fakerframe{fidx}.jpg", fakerframe)
                
                # Effectively fakerprint = reduce(sum, fakerframes)
                if fakerprint is None: fakerprint = fakerframe.astype(np.uint16)
                else: fakerprint = fakerprint + fakerframe # faster than cv2.add
            #}

            fidx += 1
        #}

        video.release()
        orig.release()

        # Scale pixel values (usually scale down to fill [0,255] range)
        assert fakerprint is not None, f"OpenCV read failure, video: {vname}"
        fakerprint = cv2.scaleAdd(fakerprint, 255/fakerprint.max(), zblock.astype(np.uint16))
        cv2.imwrite(f"{datapath}/fakerprint.jpg", fakerprint)
        return True
    #}
#}
