"""Converts Apolloscape car instance (pose and shape) data to TFRecord file format with Example protos.

The Apolloscape dataset is expected to have the following directory structure:

+ apolloscape
  + 3d_car_instance_sample
     + camera
     + car_models //.pkl file of car models
     + car_poses //json file annotation of pose and shape for cars in each image
     + images
     + split
     + pose_maps //pose map files generated by Apolloscape toolkit (http://yq01-sys-hic-k40-0003.yq01.baidu.com:8888/notebooks/baidu/personal-code/car-fitting/rui_modelfitting/dataset-api/car_instance/demo.ipynb)

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import glob
import math
import os.path
import re
import sys
import build_data
import tensorflow as tf
import numpy as np
import ntpath

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('apolloscape_root',
                           './apolloscape/3d_car_instance_sample',
                           'Apolloscape dataset root folder.')

tf.app.flags.DEFINE_string(
    'output_dir',
    './apolloscape/3d_car_instance_sample/tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')

tf.app.flags.DEFINE_string(
    'splits_dir',
    './apolloscape/3d_car_instance_sample/split',
    'Path to splits files (.txt) for train/val.')

_NUM_SHARDS = 10

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'pose_maps',
    'label': 'pose_maps',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_rescaled',
    'label': '_posemap',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'jpg',
    'label': 'npy',
}

# Image file pattern.
_IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP['image'])

def _get_files(data, dataset_split):
  """Gets files for the specified data type and dataset split.

  Args:
    data: String, desired data ('image' or 'label').
    dataset_split: String, dataset split ('train', 'val', 'test')

  Returns:
    A list of sorted file names or None when getting label for
      test set.
  """

  text_file = open('%s/%s.txt'%(FLAGS.splits_dir, dataset_split), "r")
  filenames_split = [line.replace('.jpg', '') for line in text_file.read().split('\n') if '.jpg' in line]

  if data == 'label' and dataset_split == 'test':
    return None
  pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
  search_files = os.path.join(
      FLAGS.apolloscape_root, _FOLDERS_MAP[data], pattern)
  filepaths = glob.glob(search_files)
  filepaths_split = [filepath for filepath in filepaths if ntpath.basename(filepath).replace('%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data]), '') in filenames_split]
  print dataset_split, pattern, len(filepaths_split)

  return sorted(filepaths_split)


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, val).

  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  image_files = _get_files('image', dataset_split)
  label_files = _get_files('label', dataset_split)
  num_images = len(image_files)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpg', channels=3)
  # label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        dataset_split, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(FLAGS.output_dir, shard_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_data = tf.gfile.FastGFile(image_files[i], 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        # seg_data = tf.gfile.FastGFile(label_files[i], 'rb').read()
        posemap_data = np.load(label_files[i])
        # print seg_data.shape, seg_data.dtype
        # seg_height, seg_width = label_reader.read_image_dims(seg_data)
        posemap_height, posemap_width = posemap_data.shape[0], posemap_data.shape[1]
        if height != posemap_height or width != posemap_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        re_match = _IMAGE_FILENAME_RE.search(image_files[i])
        if re_match is None:
          raise RuntimeError('Invalid image filename: ' + image_files[i])
        filename = os.path.basename(re_match.group(1))
        example = build_data.image_posemap_to_tfexample(
            image_data, filename, height, width, posemap_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  # Only support converting 'train' and 'val' sets for now.
  for dataset_split in ['train', 'val']:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
