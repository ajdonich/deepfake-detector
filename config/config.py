##########################################################################################
# 
# Varibles below that need to be set appropriately for data configuration:
#
#   1) DATA_SOURCE: must be set to either 'sample' or 'production'
#        differentiates b/w the sample kaggle dataset 
#        and the full size 50-partition kaggle dataset
#   2) DATA_LAKE_SAMP: point to the root of directory of the sample dataset
#   3) DATA_LAKE_PROD: point to the root of the full-size, partitioned dataset
#   4) COOKIES_TXT: point to a cookies.txt file output from a logged-in kaggle session
#
##########################################################################################

# Pick one:

DATA_SOURCE = 'production'
#DATA_SOURCE = 'sample'

# Set each of these:
DATA_LAKE_SAMP = '/home/ec2-user/SageMaker/ebs/deepfake-sample-test'
DATA_LAKE_PROD = '/home/ec2-user/SageMaker/ebs/deepfake-detect-datalake'
COOKIES_TXT = '../security/cookies.txt'

# ------------------ PostgreSQL configuration ----------------------

# HOST = 'posgres-free-tier.ckkzihrei3jp.us-west-2.rds.amazonaws.com'
HOST = 'postgres-deepfake-db-east.cugcvlfj9hxz.us-east-1.rds.amazonaws.com'
DATABASE = 'deepfakedb'
DBUSER = 'deepfakeusr'
DBPASSWORD = 'deepfakepwd'
DBPORT = '5432'

# ------------------------ Runtime config -------------------------

# Tensorflow callback base output directories
MODEL_STORE = '/home/ec2-user/SageMaker/models'
TBOARD_LOG = '/home/ec2-user/SageMaker/tensorboard'

TARGETSZ = (631,353)    # ConvNet output/target image size
NBATCH_PER_VIDEO = 2    # Currently 60 diffs/video, 30 diffs/batch

# --------------- The rest is dynamically created ------------------

assert DATA_SOURCE == 'sample' or DATA_SOURCE == 'production', \
    "Variable DATA_SOURCE must be either 'sample' or 'production'"

# Test data location
ROOT_DATA_TEST = f'{DATA_LAKE_SAMP}/test_videos'

# Build dynamically using kaggle partition directory structure.
DATA_LAKE = DATA_LAKE_PROD
DATA_TRAIN_PART = 'dfdc_train_part_IDX'
ROOT_DATA_TRAIN = f'{DATA_LAKE}/{DATA_TRAIN_PART}'
ROOT_FAKER_FRAMES = f'{DATA_LAKE}/dfdc_frames_part_IDX'
ROOT_FRECT_FRAMES = f'{DATA_LAKE}/dfdc_frect_part_IDX'

if DATA_SOURCE == 'sample':
#{
    # Build dynamically using kaggle sample directory structure
    DATA_LAKE = DATA_LAKE_SAMP
    DATA_TRAIN_PART = 'train_sample_videos'
    ROOT_DATA_TRAIN = f'{DATA_LAKE}/{DATA_TRAIN_PART}'
    ROOT_FAKER_FRAMES = f'/Users/ajdonich/tmp/train_sample_frames'
    ROOT_FRECT_FRAMES = f'/Users/ajdonich/tmp/train_sample_frect'
#}


