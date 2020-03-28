import time, os, cv2

import numpy as np
import tensorflow as tf

import config.config as dfc
import deepfake.dfutillib as df
import deepfake.postgresdb as pgdb


class PgdbWrapupCb(tf.keras.callbacks.Callback):
#{
    esizesql = ("SELECT COUNT(*) FROM videos WHERE "
                "blk_id = 0 AND split = 'train'")

    ingestsql = ("SELECT epoch_id FROM epoch_queue WHERE "
                 "status = 'INGESTED' ORDER BY epoch_id LIMIT 2")
    
    def __init__(self, model):
        self.model = model
        self.einitial = None        
        self.epochsz = self._queryepochsz()
        
    def _queryepochsz(self):
        with pgdb.PostgreSqlHandle() as db_handle:
            nvids_per_epoch = db_handle.sqlquery(PgdbWrapupCb.esizesql, fetch='one')[0]
            return nvids_per_epoch * dfc.NBATCH_PER_VIDEO
        
    def on_train_batch_end(self, batch, logs=None):
        nbatch = batch + 1
        if nbatch == 5 or nbatch % (self.epochsz // 5) == 0:               
            deltatime = time.time() - self.einitial
            remaintime = deltatime / nbatch * (self.epochsz - nbatch)
            print(f'{nbatch}/{self.epochsz} batches trained. Time past:',
                  f'{deltatime/60:.2f} min, ETA remain: {remaintime/60:.2f}',
                  f'min, avg/batch: {deltatime/nbatch:.2f} sec')
    
    def on_epoch_begin(self, epoch, logs=None):
        self.einitial = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
    #{
        # Complete last two ingested epochs
        with pgdb.PostgreSqlHandle() as db_handle:
            for eid in db_handle.sqlquery(PgdbWrapupCb.ingestsql, fetch='all'):
                db_handle.sqlquery(pgdb.EpochTuple.updateproto.format('COMPLETE', eid[0]))
                
        # Check for halt flag at epoch wrappup
        with pgdb.PostgreSqlHandle() as db_handle:
            if (db_handle.sqlquery(pgdb.EpochTuple.haltsql, fetch='one')[0] > 0):
                self.model.stop_training = True
    #}
#}    
        
class ModelLoader:
#{
    dequeueproto = ("SELECT * FROM epoch_queue WHERE status = 'QUEUED' "
                    "AND split = '{}' ORDER BY epoch_id LIMIT 1")

    videoproto = ("SELECT vidname, partition, label FROM videos WHERE blk_id = "
                  "{} AND split = '{}' ORDER BY video_id DESC")

    def __init__(self, split, eventloop=2):
        self.split = split
        self.eventloop = eventloop
        self.epochtuple = None
        self.videostack = []
        
        self.epochsz = self._queryepochsz()

    def _queryepochsz(self):
        sql = (f"SELECT COUNT(*) FROM videos WHERE "
               f"blk_id = 0 AND split = '{self.split}'")
        
        with pgdb.PostgreSqlHandle() as db_handle:
            nvids_per_epoch = db_handle.sqlquery(sql, fetch='one')[0]
            return nvids_per_epoch * dfc.NBATCH_PER_VIDEO

    def dequeuesql(self):
        return ModelLoader.dequeueproto.format(self.split)

    # Returns infinite generator that yields 30-frame batch per next()
    def lazy_loader(self):
    #{
        while True:
        #{
            #initial = time.time()
            videoname, partition, label = self._popvideo()
            xloader = self._loadvideo(f"{df.traindir(partition)}/{videoname}")
            yloader = self._loadtarget(f"{df.fakerdir(partition)}/{videoname[:-4]}", label)

            for x_train, (y_train, ylabels) in zip(xloader, yloader): 
                yield x_train, {'dflat_output': y_train, 'fake_output': ylabels}
            
            #print(f"Batch {vidx} processing time: {time.time()-initial:.3f} sec")
        #}
    #}

    def _popvideo(self):
        if not self.videostack:
            self.epochtuple = None
            self._loadstack()
        return self.videostack.pop()

    def _loadstack(self):
    #{
        print(f"MLoaderID: {id(self)}-{self.split}, loading blk_id: ", end='')
        
        while True:
        #{
            # Poll epoch_queue
            with pgdb.PostgreSqlHandle() as db_handle:
                eresult = db_handle.sqlquery(self.dequeuesql(), fetch='one')
                if eresult is not None: 
                    self.epochtuple = pgdb.EpochTuple(*eresult)
                    db_handle.sqlquery(self.epochtuple.updatesql('INGESTED'))
                    print(self.epochtuple.blk_id)
                        
            # Sleep or fill videostack with epoch block videos
            if self.epochtuple is None: time.sleep(self.eventloop)
            else:
                with pgdb.PostgreSqlHandle() as db_handle:
                    videosql = ModelLoader.videoproto.format(self.epochtuple.blk_id, self.split)
                    self.videostack = db_handle.sqlquery(videosql, fetch='all')
                    return self.videostack
        #}
    #}

    def _loadvideo(self, videoname, nsamps=30, nframes=60, crop=(1280,720), scsize=(640,360)):
    #{
        video = cv2.VideoCapture(videoname)

        # Calc entry frame to center capture window
        fps = round(video.get(cv2.CAP_PROP_FPS))
        fentry = round((fps*10/2) - (nframes/2))
        video.set(cv2.CAP_PROP_POS_FRAMES, fentry)

        # Get orientation from video header
        fwidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        is_portrait_orient, lft, rht, bot = df.crop_params(fwidth, fheight, crop)
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

    def _loadtarget(self, vfakerdir, label, nsamps=30, nframes=60, targsize=dfc.TARGETSZ):
    #{
        fidx, flatsz = 0, (targsize[1] * targsize[0] * 3)
        labels = (np.ones((nsamps, 1), dtype=np.uint8) if label is 'FAKE'
                  else np.zeros((nsamps, 1), dtype=np.uint8))

        while fidx < nframes:
        #{
            targets = np.zeros((nsamps, flatsz), dtype=np.uint8)
            if os.path.isdir(vfakerdir):
                for i in range(nsamps):
                    targets[i,:] = cv2.imread(f"{vfakerdir}/fakerframe{fidx+i}.jpg").flatten()

            yield targets, labels
            fidx += nsamps
        #}
    #}
#}

# class ModelLoader:
# #{
#     dequeueproto = ("SELECT * FROM epoch_queue WHERE status = 'QUEUED' "
#                     "AND split = '{}' ORDER BY epoch_id LIMIT 1")

#     videoproto = ("SELECT vidname, partition FROM videos WHERE blk_id = "
#                   "{} AND split = '{}' ORDER BY video_id DESC")

#     def __init__(self, split, eventloop=2):
#         self.split = split
#         self.eventloop = eventloop
#         self.epochtuple = None
#         self.videostack = []
        
#         self.epochsz = self._queryepochsz()

#     def _queryepochsz(self):
#         sql = (f"SELECT COUNT(*) FROM videos WHERE "
#                f"blk_id = 0 AND split = '{self.split}'")
        
#         with pgdb.PostgreSqlHandle() as db_handle:
#             nvids_per_epoch = db_handle.sqlquery(sql, fetch='one')[0]
#             return nvids_per_epoch * dfc.NBATCH_PER_VIDEO

#     def dequeuesql(self):
#         return ModelLoader.dequeueproto.format(self.split)

#     # Returns infinite generator that yields 30-frame batch per next()
#     def lazy_loader(self):
#     #{
#         while True:
#         #{
#             #initial = time.time()
#             videoname, partition = self._popvideo()
#             xloader = self._loadvideo(f"{df.traindir(partition)}/{videoname}")
#             yloader = self._loadtarget(f"{df.fakerdir(partition)}/{videoname[:-4]}")
#             for x_train, y_train in zip(xloader, yloader): yield x_train, y_train
#             #print(f"Batch {vidx} processing time: {time.time()-initial:.3f} sec")
#         #}
#     #}

#     def _popvideo(self):
#         if not self.videostack:
#             self.epochtuple = None
#             self._loadstack()
#         return self.videostack.pop()
    
#     def _loadstack(self):
#     #{
#         print(f"MLoaderID: {id(self)}-{self.split}, loading blk_id: ", end='')
        
#         while True:
#         #{
#             # Poll epoch_queue
#             with pgdb.PostgreSqlHandle() as db_handle:
#                 eresult = db_handle.sqlquery(self.dequeuesql(), fetch='one')
#                 if eresult is not None: 
#                     self.epochtuple = pgdb.EpochTuple(*eresult)
#                     db_handle.sqlquery(self.epochtuple.updatesql('INGESTED'))
#                     print(self.epochtuple.blk_id)
                        
#             # Sleep or fill videostack with epoch block videos
#             if self.epochtuple is None: time.sleep(self.eventloop)
#             else:
#                 with pgdb.PostgreSqlHandle() as db_handle:
#                     videosql = ModelLoader.videoproto.format(self.epochtuple.blk_id, self.split)
#                     self.videostack = db_handle.sqlquery(videosql, fetch='all')
#                     return self.videostack
#         #}
#     #}

#     def _loadvideo(self, videoname, nsamps=30, nframes=60, crop=(1280,720), scsize=(640,360)):
#     #{
#         video = cv2.VideoCapture(videoname)

#         # Calc entry frame to center capture window
#         fps = round(video.get(cv2.CAP_PROP_FPS))
#         fentry = round((fps*10/2) - (nframes/2))
#         video.set(cv2.CAP_PROP_POS_FRAMES, fentry)

#         # Get orientation from video header
#         fwidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#         fheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         is_portrait_orient, lft, rht, bot = df.crop_params(fwidth, fheight, crop)
#         scorientsize = (scsize[1], scsize[0]) if is_portrait_orient else scsize

#         fidx = 0
#         while video.isOpened() and fidx < nframes:
#         #{
#             samples = np.empty((nsamps, scsize[1], scsize[0], 3), dtype=np.uint8)
#             for i in range(nsamps):
#             #{
#                 vsuccess, videoframe = video.read()
#                 assert vsuccess, f"{videoname} frame {fentry+fidx+i} read failed"

#                 # Crop, then scale image size (and finally rotate if portrait orientation)
#                 videoframe = cv2.resize(videoframe[0:bot,lft:rht,:], scorientsize, interpolation=cv2.INTER_AREA)
#                 if is_portrait_orient: videoframe = cv2.rotate(videoframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 samples[i,:,:,:] = videoframe
#             #}

#             yield samples
#             fidx += nsamps
#         #}

#         video.release()
#     #}

#     def _loadtarget(self, vfakerdir, nsamps=30, nframes=60, targsize=dfc.TARGETSZ):
#     #{
#         fidx, flatsz = 0, (targsize[1] * targsize[0] * 3)

#         while fidx < nframes:
#         #{
#             targets = np.zeros((nsamps, flatsz), dtype=np.uint8)
#             if os.path.isdir(vfakerdir):
#                 for i in range(nsamps):
#                     targets[i,:] = cv2.imread(f"{vfakerdir}/fakerframe{fidx+i}.jpg").flatten()

#             yield targets
#             fidx += nsamps
#         #}
#     #}
# #}