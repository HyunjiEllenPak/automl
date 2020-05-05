"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# from generators.common import Generator
import cv2
import numpy as np
from PIL import Image
from six import raise_from
import csv
import sys
import os.path as osp
from collections import OrderedDict
import os

def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """
    Parse the classes file given by csv_reader.
    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_quadrangle_annotations(csv_reader, classes, detect_text=False):
    """
    Read annotations from the csv_reader.
    Args:
        csv_reader: csv reader of args.annotations_path
        classes: list[str] all the class names read from args.classes_path

    Returns:
        result: dict, dict is like {image_path: [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                     'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4, 'class': class_name}]}

    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader, 1):
        try:
            img_file, x1, y1, x2, y2, x3, y3, x4, y4, class_name = row[:10]
            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, x3, y3, x4, y4, class_name) == ('', '', '', '', '', '', '', '', ''):
                continue

            x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))
            x3 = _parse(x3, int, 'line {}: malformed x3: {{}}'.format(line))
            y3 = _parse(y3, int, 'line {}: malformed y3: {{}}'.format(line))
            x4 = _parse(x4, int, 'line {}: malformed x4: {{}}'.format(line))
            y4 = _parse(y4, int, 'line {}: malformed y4: {{}}'.format(line))

            # check if the current class name is correctly present
            if detect_text:
                if class_name == '###':
                    continue
                else:
                    class_name = 'text'

            if class_name not in classes:
                raise ValueError(f'line {line}: unknown class name: \'{class_name}\' (classes: {classes})')

            result[img_file].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                     'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4, 'class': class_name})
        except ValueError:
            raise_from(ValueError(
                f'line {line}: format should be \'img_file,x1,y1,x2,y2,x3,y3,x4,y4,class_name\' or \'img_file,,,,,\''),
                None)

    return result


def _read_annotations(csv_reader, classes, base_dir):
    """
    Read annotations from the csv_reader.
    Args:
        csv_reader: csv reader of args.annotations_path
        classes: list[str] all the class names read from args.classes_path

    Returns:
        result: dict, dict is like {image_path: [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_name}]}

    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader, 1):
        try:
            img_file, x1, y1, x2, y2 = row[:5]
            class_name = img_file.split("/")[0]
            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            if class_name not in classes:
                raise ValueError(f'line {line}: unknown class name: \'{class_name}\' (classes: {classes})')

            result[img_file].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_name,
                                     'filename':img_file})
        except ValueError:
            raise_from(ValueError(
                f'line {line}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''),
                None)

    return result


def _open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb', for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')

def load_image(path):
     """
     Load an image at the image_index.
     """
     image = cv2.imread(path)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     return image