###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Visual Wake Word Dataset (using pyvww)
"""
import os

import torchvision
from torchvision import transforms
import pyvww

import ai8x


def vww_get_datasets(data, load_train=True, load_test=True, input_size=224, coco_folder="COCO/all2014"):
    """
    Load the Visual Wake Word Person Classification dataset.

    The original training dataset is split into training and validation sets.
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-128/128, +127/128]

    Data augmentation: 4 pixels are padded on each side, and a 224x224 crop is randomly sampled
    from the padded image or its horizontal flip.
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ai8x.normalize(args=args),
        ])

        train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
            root=data_dir + '/' + coco_folder,
            annFile=data_dir + "/vww/annotations/instances_train.json",
            transform=train_transform,
        )

    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ai8x.normalize(args=args),
        ])

        test_dataset = pyvww.pytorch.VisualWakeWordsClassification(
            root=data_dir + '/' + coco_folder,
            annFile=data_dir + "/vww/annotations/instances_val.json",
            transform=test_transform,
        )

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def vwwfolder_get_datasets(data, load_train=True, load_test=True, input_size=224):
    """
    Load the Visual Wake Word (COCO 2014) Classification dataset using ImageFolder.
    _This function is used when the number of output classes is less than the default and
    it depends on a custom installation._
    """
    return vww_get_datasets(data, load_train, load_test, input_size)


datasets = [
    {
        'name': 'VisualWakeWord',
        'input': (3, 224, 224),
        'output': [0,1],
        'loader': vww_get_datasets,
    },
]
