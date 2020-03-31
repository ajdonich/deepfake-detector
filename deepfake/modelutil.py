import time, os, re, cv2

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config.config as dfc
import deepfake.dfutillib as df
import deepfake.postgresdb as pgdb


class ModelWrapper:
#{
    def __init__(self):
        self.model = None
        self.diff_output = None
        self.fake_output = None
        
    def init_encdec_network(self, traindiff=True, trainfake=True):
    #{
        # Layer #0: Input
        frame_input = keras.layers.Input(shape=(360, 640, 3))

        # Layer #1: First set of convolutional layers
        X = keras.layers.Conv2D(filters=32, kernel_size=(11,11), 
                                strides=(2,4), trainable=traindiff)(frame_input)
        X = keras.layers.MaxPooling2D()(X)

        # Layer #2: Second set of convolutional layers
        X = keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                                strides=(1,1), trainable=traindiff)(X)
        X = keras.layers.MaxPooling2D()(X)

        # Layer #3: Third set of convolutional layers
        X = keras.layers.Conv2D(filters=128, kernel_size=(3,3), 
                                strides=(1,1), trainable=traindiff)(X)
        X = keras.layers.MaxPooling2D()(X)

        # Layer #4-5: Fully connected layers
        X_flat = keras.layers.Flatten()(X)
        X = keras.layers.Dense(720, activation = "relu", trainable=traindiff)(X_flat)
        X = keras.layers.Dense(720, activation = "relu", trainable=traindiff)(X)
        X = keras.layers.Concatenate(axis=1)([X, X_flat])
        X = keras.layers.Reshape((20,18,130))(X)

        # Layer #6: First set of deconvolutional layers
        X = keras.layers.UpSampling2D()(X)
        X = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), 
                                         strides=(1,1), trainable=traindiff)(X)

        # Layer #7: Second set of deconvolutional layers
        X = keras.layers.UpSampling2D()(X)
        X = keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), 
                                         strides=(1,1), trainable=traindiff)(X)

        # Layer #8: Last deconvolution to diff image output
        X = keras.layers.UpSampling2D()(X)
        diff_output = keras.layers.Conv2DTranspose(filters=3, kernel_size=(11,11), 
                                                   strides=(2,4), trainable=traindiff)(X)

        # Layer #9-10: Final fully connected to binary REAL/FAKE class
        X = keras.layers.MaxPooling2D(pool_size=(6, 6))(diff_output)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(512, activation = "relu", trainable=trainfake)(X)
        X = keras.layers.Dense(512, activation = "relu", trainable=trainfake)(X)

        # Output layers
        dflat_output = keras.layers.Flatten(name='dflat_output')(diff_output)
        fake_output = keras.layers.Dense(1, activation = "sigmoid",
                                         trainable=trainfake, name='fake_output')(X)

        if not traindiff: self.model = keras.models.Model(inputs=frame_input, outputs=fake_output)
        else: self.model = keras.models.Model(inputs=frame_input, outputs=[dflat_output, fake_output])
        self.model.summary()
        
        self.diff_output = diff_output
        self.fake_output = fake_output
        return self.model
    #}
#}

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

    def __init__(self, split, traindiff=True, eventloop=2):
        self.split = split
        self.traindiff = traindiff
        self.eventloop = eventloop
        self.epochtuple = None
        self.videostack = []
        
        self.epochsz = self._queryepochsz()

    def _queryepochsz(self):
    #{
        if self.split == 'test':
            mp4 = lambda vid: re.match(r'\w+\.mp4', vid)
            self.videostack = sorted(filter(mp4, os.listdir(dfc.ROOT_DATA_TEST)))
            return len(self.videostack)
        
        elif self.split == 'holdout':
            self.videostack = self._loadholdout()
            return len(self.videostack)

        else: 
            sql = (f"SELECT COUNT(*) FROM videos WHERE "
                   f"blk_id = 0 AND split = '{self.split}'")

            with pgdb.PostgreSqlHandle() as db_handle:
                nvids_per_epoch = db_handle.sqlquery(sql, fetch='one')[0]
                return nvids_per_epoch * dfc.NBATCH_PER_VIDEO
    #}
    
    def dequeuesql(self):
        return ModelLoader.dequeueproto.format(self.split)

    # Returns infinite generator that yields 30-frame batch per next()
    def lazy_loader(self):
    #{
        while True:
        #{
            videoname, partition, label = self._popvideo()
            xloader = self._loadvideo(f"{df.traindir(partition)}/{videoname}")
            yloader = self._loadtarget(f"{df.fakerdir(partition)}/{videoname[:-4]}", label)

            if self.traindiff:
                for x_train, (y_train, ylabels) in zip(xloader, yloader): 
                    yield x_train, {'dflat_output': y_train, 'fake_output': ylabels}
            else: 
                for x_train, ylabels in zip(xloader, yloader): yield x_train, ylabels
        #}
    #}
    
    def holdout_loader(self):
    #{
        for videoname, partition, label in self.videostack:
            videoname = f"{df.traindir(partition)}/{videoname}"
            for x_holdout in self._loadvideo(videoname, nsamps=60):
                yield x_holdout
    #}
    
    def test_loader(self):
    #{
        for vidx, videoname in enumerate(self.videostack):
        #{
            #initial = time.time()
            videoname = f"{dfc.ROOT_DATA_TEST}/{videoname}"
            for x_test in self._loadvideo(videoname, nsamps=60): 
                yield x_test
            
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
                if eresult is None: self._reinsertblock(); continue
                else: 
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

    def _loadholdout(self):
    #{
        print(f"MLoaderID: {id(self)}-{self.split}, loading blk_id: -1")
        
        with pgdb.PostgreSqlHandle() as db_handle:
            videosql = ModelLoader.videoproto.format(-1, self.split)
            self.videostack = db_handle.sqlquery(videosql, fetch='all')
            return self.videostack
    #}
    
    def _reinsertblock(self):
    #{
        blk_id = None
        unpack = lambda tup: tup[0]
        sql = "SELECT blk_id FROM epoch_queue where split='train'"
        with pgdb.PostgreSqlHandle() as db_handle:
            blk_id = np.random.choice([blkid for blkid in map(
                unpack, db_handle.sqlquery(sql, fetch='all'))])

        # Insert epoch block onto the queue
        with pgdb.PostgreSqlHandle() as db_handle:
            etraintup = pgdb.EpochTuple(blk_id=blk_id, split='train', status='QUEUED')
            evalidtup = pgdb.EpochTuple(blk_id=blk_id, split='validate', status='QUEUED')
            db_handle.sqlquery(etraintup.insertsql()); db_handle.sqlquery(evalidtup.insertsql())
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
        labels = (np.ones((nsamps, 1), dtype=np.float32) if label == 'FAKE'
                  else np.zeros((nsamps, 1), dtype=np.float32))
                
        while fidx < nframes:
        #{
            if self.traindiff:
                targets = np.zeros((nsamps, flatsz), dtype=np.uint8)
                if os.path.isdir(vfakerdir):
                    for i in range(nsamps):
                        targets[i,:] = cv2.imread(f"{vfakerdir}/fakerframe{fidx+i}.jpg").flatten()

                yield targets, labels
            else: yield labels
            fidx += nsamps
        #}
    #}
#}

