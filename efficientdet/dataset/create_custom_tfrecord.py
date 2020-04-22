# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert PASCAL dataset to TFRecord.

Example usage:
    python create_pascal_tfrecord.py  --data_dir=/tmp/VOCdevkit  \
        --year=VOC2012  --output_path=/tmp/pascal
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf

from dataset import tfrecord_util
from dataset.csv_ import _read_classes, _open_for_csv, raise_from, _read_annotations
import csv

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to custom dataset.')
flags.DEFINE_string('set', 'train', 'train or validation or test.')
# flags.DEFINE_string('annotations_dir', 'Annotations',
#                     '(Relative) path to annotations directory.')

flags.DEFINE_string('output_path', '', 'Path to output TFRecord and json.')
# flags.DEFINE_string('csv_class_path', None,
#                     'Path to class mapping file with p json file with a dictionary.')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_integer('num_shards', 1, 'Number of shards for output file.')
flags.DEFINE_integer('num_images', None, 'Max number of imags to process.')
FLAGS = flags.FLAGS

POSE = 'unspecified'

GLOBAL_IMG_ID = 0  # global image id.
GLOBAL_ANN_ID = 0  # global annotation id.


def get_image_id(filename):
  """Convert a string to a integer."""
  # Warning: this function is highly specific to pascal filename!!
  # Given filename like '2008_000002', we cannot use id 2008000002 because our
  # code internally will convert the int value to float32 and back to int, which
  # would cause value mismatch int(float32(2008000002)) != int(2008000002).
  # COCO needs int values, here we just use a incremental global_id, but
  # users should customize their own ways to generate filename.
  del filename
  global GLOBAL_IMG_ID
  GLOBAL_IMG_ID += 1
  return GLOBAL_IMG_ID


def get_ann_id():
  """Return unique annotation id across images."""
  global GLOBAL_ANN_ID
  GLOBAL_ANN_ID += 1
  return GLOBAL_ANN_ID


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages',
                       ann_json_dict=None):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running tfrecord_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.
    ann_json_dict: annotation json dictionary.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  # img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])

  full_path = os.path.join(dataset_directory,data['filename'])
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)

  image = PIL.Image.open(encoded_jpg_io)

  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  # width = int(data['size']['width'])
  # height = int(data['size']['height'])
  width, height = image.size
  image_id = get_image_id(data['filename'])
  if ann_json_dict:
    image = {
        'file_name': data['filename'],
        'height': height,
        'width': width,
        'id': image_id,
    }
    ann_json_dict['images'].append(image)

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  # if 'object' in data:
  #   for obj in data['object']:
  # difficult = bool(int(obj['difficult']))
  difficult = bool(0)
  # if ignore_difficult_instances and difficult:
  #     continue

  difficult_obj.append(int(difficult))

  xmin.append(float(data['x1']) / width)
  ymin.append(float(data['y1']) / height)
  xmax.append(float(data['x2']) / width)
  ymax.append(float(data['y2']) / height)

  classes_text.append(data['class'].encode('utf8'))
  classes.append(label_map_dict[data['class']])
  truncated.append(int(0))
  poses.append(POSE.encode('utf8'))

  if ann_json_dict:
      abs_xmin = int(data['xmin'])
      abs_ymin = int(data['ymin'])
      abs_xmax = int(data['xmax'])
      abs_ymax = int(data['ymax'])
      abs_width = abs_xmax - abs_xmin
      abs_height = abs_ymax - abs_ymin
      ann = {
          'area': abs_width * abs_height,
          'iscrowd': 0,
          'image_id': image_id,
          'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
          'category_id': label_map_dict[data['class']],
          'id': get_ann_id(),
          'ignore': 0,
          'segmentation': [],
      }
      ann_json_dict['annotations'].append(ann)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': tfrecord_util.int64_feature(height),
      'image/width': tfrecord_util.int64_feature(width),
      'image/filename': tfrecord_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': tfrecord_util.bytes_feature(
          str(image_id).encode('utf8')),
      'image/key/sha256': tfrecord_util.bytes_feature(key.encode('utf8')),
      'image/encoded': tfrecord_util.bytes_feature(encoded_jpg),
      'image/format': tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': tfrecord_util.float_list_feature(xmin),
      'image/object/bbox/xmax': tfrecord_util.float_list_feature(xmax),
      'image/object/bbox/ymin': tfrecord_util.float_list_feature(ymin),
      'image/object/bbox/ymax': tfrecord_util.float_list_feature(ymax),
      'image/object/class/text': tfrecord_util.bytes_list_feature(classes_text),
      'image/object/class/label': tfrecord_util.int64_list_feature(classes),
      'image/object/difficult': tfrecord_util.int64_list_feature(difficult_obj),
      'image/object/truncated': tfrecord_util.int64_list_feature(truncated),
      'image/object/view': tfrecord_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))
  if not FLAGS.output_path:
    raise ValueError('output_path cannot be empty.')

  data_dir = FLAGS.data_dir
  logging.info('writing to output path: %s', FLAGS.output_path)
  writers = [
      tf.python_io.TFRecordWriter(
          FLAGS.output_path + '-%05d-of-%05d.tfrecord' % (i, FLAGS.num_shards))
      for i in range(FLAGS.num_shards)
  ]

  csv_class_file = os.path.join(data_dir, 'class_mapping.csv')
  try:
      with _open_for_csv(csv_class_file) as file:
          # class_name --> class_id
          classes = _read_classes(csv.reader(file, delimiter=','))
  except ValueError as e:
      raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)
  label_map_dict = {}
  # class_id --> class_name
  for key, value in classes.items():
      label_map_dict[value] = key

  ann_json_dict = {
      'images': [],
      'type': 'instances',
      'annotations': [],
      'categories': []
  }

  for class_name, class_id in label_map_dict.items():
      cls = {'supercategory': 'none', 'id': class_id, 'name': class_name}
      ann_json_dict['categories'].append(cls)

  logging.info('Reading from Custom dataset')

  csv_data_file = os.path.join(data_dir, FLAGS.set)
  try:
      with _open_for_csv(csv_data_file) as file:
          # {'img_path1':[{'x1':xx,'y1':xx,'x2':xx,'y2':xx,'x3':xx,'y3':xx,'x4':xx,'y4':xx, 'class':xx}...],...}
          image_dataset = _read_annotations(csv.reader(file, delimiter=','), classes)
  except ValueError as e:
      raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)

  for idx, imagedata in enumerate(image_dataset):
      tf_example = dict_to_tf_example(image_dataset, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances,
                                      ann_json_dict=ann_json_dict)
      writers[idx % FLAGS.num_shards].write(tf_example.SerializeToString())

  for writer in writers:
    writer.close()

  json_file_path = os.path.join(
      os.path.dirname(FLAGS.output_path),
      'json_' + os.path.basename(FLAGS.output_path) + '.json')
  with tf.io.gfile.GFile(json_file_path, 'w') as f:
    json.dump(ann_json_dict, f)


if __name__ == '__main__':
  tf.app.run()
