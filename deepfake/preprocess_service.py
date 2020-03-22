import time, re, os
import cv2
import numpy as np

import config.config as dfc
import deepfake.dfutillib as df
import deepfake.postgresdb as pgdb

class Preprocessor:
#{
    def __init__(self, minqueued=5, eventloop=10, logbatch=50):
        self.minqueued = minqueued
        self.eventloop = eventloop
        self.logbatch = logbatch
        self.maxblkid = self._queryblkid()

    def _queryblkid(self):
        sql = "SELECT MAX(blk_id) FROM videos"
        with pgdb.PostgreSqlHandle() as db_handle:
            return db_handle.sqlquery(sql)[0]

    def run(self):
    #{
        unpackfcn = lambda vt: pgdb.VideoTuple(*vt)
        countsql = ("SELECT COUNT(*) FROM epoch_queue WHERE "
                    "status = 'QUEUED' OR status = 'RUNNING'")

        haltflag = False
        haltsql = ("SELECT COUNT(*) FROM epoch_queue WHERE status = 'HALT'")
        with pgdb.PostgreSqlHandle() as db_handle:
            haltflag = (db_handle.sqlquery(haltsql)[0] > 0)

        while not haltflag:
        #{
            doappend = False
            with pgdb.PostgreSqlHandle() as db_handle:
                doappend = (db_handle.sqlquery(countsql)[0] < self.minqueued)

            if not doappend: time.sleep(self.eventloop)
            else:
            #{
                # Read unprocessed tuples from video table
                blk_id = np.random.randint(self.maxblkid+1)
                trainsql = (f"SELECT * FROM videos WHERE blk_id = {blk_id} AND "
                            f"split = 'train' AND proc_flg = FALSE AND label = 'FAKE'")
                validsql = (f"SELECT * FROM videos WHERE blk_id = {blk_id} AND "
                            f"split = 'validate' AND proc_flg = FALSE AND label = 'FAKE'")

                vidtuples = []
                with pgdb.PostgreSqlHandle() as db_handle:
                    vidtuples = [vtup for vtup in map(unpackfcn, db_handle.sqlquery(trainsql))]
                    vidtuples += [vtup for vtup in map(unpackfcn, db_handle.sqlquery(validsql))]
                
                videoids = []
                totproctime = time.time()
                print(f"Preprocess epoch block {blk_id}:")
                for vtup in vidtuples:
                #{
                    # Create and save diff images
                    proctimes.append(time.time())
                    vdir = re.split(r'[/.]', vtup.vidname)[-2]
                    datapath = f"{df.fakerdir(vtup.part_id)}/{vdir}"
                    if not os.path.isdir(datapath): os.mkdir(datapath)

                    videourl = f"{df.traindir(vtup.part_id)}/{vtup.vidname}"
                    origurl = f"{df.traindir(vtup.part_id)}/{vtup.origname}"
                    self.create_diff_frames(videourl, origurl, datapath)
                    videoids.append(vtup.video_id)

                    # Logging
                    proctimes[-1] = time.time() - proctimes[-1]
                    if len(proctimes) == self.logbatch:
                        print(f"  Running average preprocess time/pair: \
                              {np.average(proctimes):.2f} sec")
                        proctimes = []
                #}
                print(f"  Total process time: {time.time() - totproctime:.2f} sec")

                # Update preprocess flag on videos 
                with pgdb.PostgreSqlHandle() as db_handle:
                    db_handle.sqlquery(f"UPDATE videos SET proc_flg = TRUE WHERE video_id IN {tuple(videoids)}")

                # Insert epoch block into queue
                # with pgdb.PostgreSqlHandle() as db_handle:
                #     db_handle.sqlquery(f"UPDATE videos SET proc_flg = TRUE WHERE video_id IN {tuple(videoids)}")
            #}

            # Check for halt flag
            with pgdb.PostgreSqlHandle() as db_handle:
                haltflag = (db_handle.sqlquery(haltsql)[0] > 0)
        #}
    #}

    # Given a video name pair: vname a deepfake, and oname its original/parent, function creates
    # "fakerframe" images (scaled-difference-blend-mode-images), cropped to size crop, starting
    # from video frame: fentry and then contiguously for: nframes. Also creates a 'fakerprint' 
    # per pair, which is just the scaled sum the fakerframes. All images files saved to datapath.
    def create_diff_frames(self, vname, oname, datapath, fentry=120, nframes=60, crop=(1280,720), targsize=(547,347)):
    #{
        video = cv2.VideoCapture(vname)
        orig = cv2.VideoCapture(oname)
        
        video.set(cv2.CAP_PROP_POS_FRAMES, fentry)
        orig.set(cv2.CAP_PROP_POS_FRAMES, fentry)

        # Get orientation from video header
        fwidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        is_portrait_orient, lft, rht, bot = df._crop_params(fwidth, fheight, crop)
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
                cv2.imwrite(f"{datapath}/fakerframe{fentry+fidx}.jpg", fakerframe)
                
                # Effectively fakerprint = reduce(sum, fakerframes)
                if fakerprint is None: fakerprint = fakerframe.astype(np.uint16)
                else: fakerprint = fakerprint + fakerframe # faster than cv2.add
            #}

            fidx += 1
        #}
        
        # Scale pixel values (usually scale down to fill [0,255] range)
        fakerprint = cv2.scaleAdd(fakerprint, 255/fakerprint.max(), zblock.astype(np.uint16))
        cv2.imwrite(f"{datapath}/fakerprint.jpg", fakerprint)
        
        video.release()
        orig.release()
    #}
#}