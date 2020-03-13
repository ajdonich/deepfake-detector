##########################################################################################
# 
# Varibles below that need to be set appropriately for configuration:
#
#   1) DATA_SOURCE: must be set to either 'sample' or 'production'
#        differentiates b/w the sample kaggle dataset 
#        and the full size 50-partition kaggle dataset
#   2) DATA_LAKE_SAMP: point to the root of directory of the sample dataset
#   3) DATA_LAKE_PROD: point to the root of the full-size, partitioned dataset
#   4) COOKIES_TXT: point to a cookies.txt file output from a logged in kaggle session
#
##########################################################################################

DATA_SOURCE = 'production'
#DATA_SOURCE = 'sample'

DATA_LAKE_SAMP = '../data'
DATA_LAKE_PROD = '/Volumes/My Book/deepfake-detect-datalake'
COOKIES_TXT = '../security/cookies.txt'

# -----------------------------------------------------------------------------------------

assert DATA_SOURCE == 'sample' or DATA_SOURCE == 'production', \
    "Env variable DATA_SOURCE_FLAG must be either 'sample' or 'production'"

# Built dynamically using kaggle partition directory structure.
DATA_LAKE = DATA_LAKE_PROD
DATA_TRAIN_PART = 'dfdc_train_part_IDX'
ROOT_DATA_TRAIN = f'{DATA_LAKE}/{DATA_TRAIN_PART}'
ROOT_FAKER_FRAMES = f'{DATA_LAKE}/dfdc_frames_part_IDX'

if DATA_SOURCE == 'sample':
#{
    # Build dynamically using kaggle sample directory structure
    DATA_LAKE = DATA_LAKE_SAMP
    DATA_TRAIN_PART = 'train_sample_videos'
    ROOT_DATA_TRAIN = f'{DATA_LAKE}/{DATA_TRAIN_PART}'
    ROOT_FAKER_FRAMES = f'{DATA_LAKE}/train_sample_frames'
#}


