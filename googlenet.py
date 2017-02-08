import numpy as np
import random
import json
import argparse
import signal
import math
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    merge, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from googlenet_custom_layers import PoolHelper, LRN


class DataGenerator:
    def __init__(self, input_data_path, batch_size=128, test_data_percent=0.1, background_signal_equivalent=True):
        print("Load Data From %s." % input_data_path)
        f = open(input_data_path, "r")
        data = json.loads(f.read())
        self._signal = data['signal']
        self._background = data['background']
        self._max_energy = None
        self._data = []
        if not background_signal_equivalent:
            print("Load %s signal, %s background" % (len(self._signal), len(self._background)))
            for s in self._signal:
                self._data.append((s, 0))
            for b in self._background:
                self._data.append((b, 1))
        else:
            size = min(len(self._signal), len(self._background))
            print("Load %s signal, %s background" % (size, size))
            for i in range(0, size):
                self._data.append((self._signal[i], 0))
            for i in range(0, size):
                self._data.append((self._background[i], 1))
        print("Totally load %s data." % len(self._data))
        random.shuffle(self._data)
        split_at = int(math.floor(len(self._data) * (1 - test_data_percent)))
        self._train_data = self._data[:split_at]
        self._test_data = self._data[split_at:]
        self._batch_size = batch_size
        self._train_pointer = 0
        self._test_pointer = 0

    def set_max_energy(self, max_energy):
        self._max_energy = max_energy

    def get_train_size(self):
        return len(self._train_data)

    def get_test_size(self):
        return len(self._test_data)

    def get_batch_size(self):
        return self._batch_size

    def _convert_row(self, input_row):
        row = np.zeros((3, 224, 224))
        cluster_xy_data = input_row[0]
        for pixel, energy in cluster_xy_data.items():
            location = pixel.split(":")
            location_x = int(location[0]) * 4
            location_y = int(location[1]) * 4
            for i in range(0, 4):
                for j in range(0, 4):
                    _location_x = location_x + 224 / 2 + i
                    _location_y = location_y + 224 / 2 + j
                    if not (0 <= _location_x < 224 and 0 <= _location_y < 224):
                        continue
                    if self._max_energy:
                        row[0, _location_x, _location_y] = min(int(math.floor(energy / self._max_energy * 256)), 255)
                    else:
                        row[0, _location_x, _location_y] = min(int(math.floor(energy / input_row[2] * 256)), 255)
        cluster_zy_data = input_row[1]
        for pixel, energy in cluster_zy_data.items():
            location = pixel.split(":")
            location_z = int(location[0]) * 4
            location_y = int(location[1]) * 4
            for i in range(0, 4):
                for j in range(0, 4):
                    _location_z = location_z + 224 / 2 + i
                    _location_y = location_y + 224 / 2 + j
                    if not (0 <= _location_z < 224 and 0 <= _location_y < 224):
                        continue
                    if self._max_energy:
                        row[1, _location_z, _location_y] = min(int(math.floor(energy / self._max_energy * 256)), 255)
                    else:
                        row[1, _location_z, _location_y] = min(int(math.floor(energy / input_row[2] * 256)), 255)
        return row

    def train_generator(self):
        while True:
            start = self._train_pointer
            end = self._train_pointer + self._batch_size
            if end >= len(self._train_data):
                end = len(self._train_data)
                self._train_pointer = 0
            else:
                self._train_pointer = end
            data = self._train_data[start:end]
            count = len(data)
            result_x = np.zeros((count, 3, 224, 224), dtype='float32')
            result_y = np.zeros((count, 1000))
            for i, row in enumerate(data):
                result_x[i] = self._convert_row(row[0])
                result_y[i][row[1]] = 1
            yield result_x, [result_y, result_y, result_y]

    def test_generator(self):
        while True:
            start = self._test_pointer
            end = self._test_pointer + self._batch_size
            if end >= len(self._test_data):
                end = len(self._test_data)
                self._test_pointer = 0
            else:
                self._test_pointer = end
            data = self._test_data[start:end]
            count = len(data)
            result_x = np.zeros((count, 3, 224, 224), dtype='float32')
            result_y = np.zeros((count, 1000))
            for i, row in enumerate(data):
                result_x[i] = self._convert_row(row[0])
                result_y[i][row[1]] = 1
            yield result_x, [result_y, result_y, result_y]

    def get_some_test(self, size):
        result_x = np.zeros((size, 3, 224, 224), dtype='float32')
        result_y = np.zeros((size, 1000))
        for i in range(0, size):
            row = random.choice(self._test_data)
            result_x[i] = self._convert_row(row[0])
            result_y[i][row[1]] = 1
        return result_x, result_y


class TestDataGenerator:
    def __init__(self, count=100000, batch_size=128, test_data_percent=0.1):
        self._data = []
        self._batch_size = batch_size
        self.test_data_percent = test_data_percent
        self._type = ["circle", "square"]
        self._train_pointer = 0
        self._test_pointer = 0
        for i in range(0, count):
            type_ = random.choice(self._type)
            x = random.randint(0, 223)
            y = random.randint(0, 223)
            r = random.randint(0, 50)
            self._data.append((type_, x, y, r))
        random.shuffle(self._data)
        split_at = int(math.floor(len(self._data) * (1 - test_data_percent)))
        self._train_data = self._data[:split_at]
        self._test_data = self._data[split_at:]

    def get_train_size(self):
        return len(self._train_data)

    def get_test_size(self):
        return len(self._test_data)

    def _convert_row(self, input_row):
        t_, x_, y_, r_ = input_row
        row = np.zeros((3, 224, 224))
        if t_ == "circle":
            for i in range(0, 224):
                for j in range(0, 224):
                    if math.sqrt((i - x_) * (i - x_) + (j - y_) * (j - y_)) < r_:
                        row[0, i, j] = 255
        elif t_ == "square":
            for i in range(0, 224):
                for j in range(0, 224):
                    if abs(i - x_) < r_ and abs(j - y_) < r_:
                        row[0, i, j] = 255
        return row

    def train_generator(self):
        while True:
            start = self._train_pointer
            end = self._train_pointer + self._batch_size
            if end >= len(self._train_data):
                end = len(self._train_data)
                self._train_pointer = 0
            else:
                self._train_pointer = end
            data = self._train_data[start:end]
            count = len(data)
            result_x = np.zeros((count, 3, 224, 224), dtype='float32')
            result_y = np.zeros((count, 1000))
            for i, row in enumerate(data):
                result_x[i] = self._convert_row(row)
                if row[0] == "circle":
                    result_y[i][0] = 1
                else:
                    result_y[i][1] = 1
            yield result_x, [result_y, result_y, result_y]

    def test_generator(self):
        while True:
            start = self._test_pointer
            end = self._test_pointer + self._batch_size
            if end >= len(self._test_data):
                end = len(self._test_data)
                self._test_pointer = 0
            else:
                self._test_pointer = end
            data = self._test_data[start:end]
            count = len(data)
            result_x = np.zeros((count, 3, 224, 224), dtype='float32')
            result_y = np.zeros((count, 1000))
            for i, row in enumerate(data):
                result_x[i] = self._convert_row(row)
                if row[0] == "circle":
                    result_y[i][0] = 1
                else:
                    result_y[i][1] = 1
            yield result_x, [result_y, result_y, result_y]

    def get_some_test(self, size):
        result_x = np.zeros((size, 3, 224, 224), dtype='float32')
        result_y = np.zeros((size, 1000))
        for i in range(0, size):
            row = random.choice(self._test_data)
            result_x[i] = self._convert_row(row)
            if row[0] == "circle":
                result_y[i][0] = 1
            else:
                result_y[i][1] = 1
        return result_x, result_y


class Train:
    def __init__(self):
        self.google_net = None
        self.exit_signal = False
        self.thread = None
        self.data_generator = None

    def set_data(self, input_data):
        # self.data_generator = DataGenerator(input_data)
        self.data_generator = TestDataGenerator()

    def create_googlenet(self, weights_path=None):
        # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)

        input = Input(shape=(3, 224, 224))

        conv1_7x7_s2 = Convolution2D(64, 7, 7, subsample=(2, 2), border_mode='same', activation='relu',
                                     name='conv1/7x7_s2',
                                     W_regularizer=l2(0.0002))(input)

        conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)

        pool1_helper = PoolHelper()(conv1_zero_pad)

        pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool1/3x3_s2')(
            pool1_helper)

        pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

        conv2_3x3_reduce = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name='conv2/3x3_reduce',
                                         W_regularizer=l2(0.0002))(pool1_norm1)

        conv2_3x3 = Convolution2D(192, 3, 3, border_mode='same', activation='relu', name='conv2/3x3',
                                  W_regularizer=l2(0.0002))(conv2_3x3_reduce)

        conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)

        conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)

        pool2_helper = PoolHelper()(conv2_zero_pad)

        pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool2/3x3_s2')(
            pool2_helper)

        inception_3a_1x1 = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name='inception_3a/1x1',
                                         W_regularizer=l2(0.0002))(pool2_3x3_s2)

        inception_3a_3x3_reduce = Convolution2D(96, 1, 1, border_mode='same', activation='relu',
                                                name='inception_3a/3x3_reduce', W_regularizer=l2(0.0002))(pool2_3x3_s2)

        inception_3a_3x3 = Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='inception_3a/3x3',
                                         W_regularizer=l2(0.0002))(inception_3a_3x3_reduce)

        inception_3a_5x5_reduce = Convolution2D(16, 1, 1, border_mode='same', activation='relu',
                                                name='inception_3a/5x5_reduce', W_regularizer=l2(0.0002))(pool2_3x3_s2)

        inception_3a_5x5 = Convolution2D(32, 5, 5, border_mode='same', activation='relu', name='inception_3a/5x5',
                                         W_regularizer=l2(0.0002))(inception_3a_5x5_reduce)

        inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_3a/pool')(
            pool2_3x3_s2)

        inception_3a_pool_proj = Convolution2D(32, 1, 1, border_mode='same', activation='relu',
                                               name='inception_3a/pool_proj', W_regularizer=l2(0.0002))(
            inception_3a_pool)

        inception_3a_output = merge([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj],
                                    mode='concat', concat_axis=1, name='inception_3a/output')

        inception_3b_1x1 = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='inception_3b/1x1',
                                         W_regularizer=l2(0.0002))(inception_3a_output)

        inception_3b_3x3_reduce = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                                name='inception_3b/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_3a_output)

        inception_3b_3x3 = Convolution2D(192, 3, 3, border_mode='same', activation='relu', name='inception_3b/3x3',
                                         W_regularizer=l2(0.0002))(inception_3b_3x3_reduce)

        inception_3b_5x5_reduce = Convolution2D(32, 1, 1, border_mode='same', activation='relu',
                                                name='inception_3b/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_3a_output)

        inception_3b_5x5 = Convolution2D(96, 5, 5, border_mode='same', activation='relu', name='inception_3b/5x5',
                                         W_regularizer=l2(0.0002))(inception_3b_5x5_reduce)

        inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_3b/pool')(
            inception_3a_output)

        inception_3b_pool_proj = Convolution2D(64, 1, 1, border_mode='same', activation='relu',
                                               name='inception_3b/pool_proj', W_regularizer=l2(0.0002))(
            inception_3b_pool)

        inception_3b_output = merge([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj],
                                    mode='concat', concat_axis=1, name='inception_3b/output')

        inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)

        pool3_helper = PoolHelper()(inception_3b_output_zero_pad)

        pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool3/3x3_s2')(
            pool3_helper)

        inception_4a_1x1 = Convolution2D(192, 1, 1, border_mode='same', activation='relu', name='inception_4a/1x1',
                                         W_regularizer=l2(0.0002))(pool3_3x3_s2)

        inception_4a_3x3_reduce = Convolution2D(96, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4a/3x3_reduce', W_regularizer=l2(0.0002))(pool3_3x3_s2)

        inception_4a_3x3 = Convolution2D(208, 3, 3, border_mode='same', activation='relu', name='inception_4a/3x3',
                                         W_regularizer=l2(0.0002))(inception_4a_3x3_reduce)

        inception_4a_5x5_reduce = Convolution2D(16, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4a/5x5_reduce', W_regularizer=l2(0.0002))(pool3_3x3_s2)

        inception_4a_5x5 = Convolution2D(48, 5, 5, border_mode='same', activation='relu', name='inception_4a/5x5',
                                         W_regularizer=l2(0.0002))(inception_4a_5x5_reduce)

        inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_4a/pool')(
            pool3_3x3_s2)

        inception_4a_pool_proj = Convolution2D(64, 1, 1, border_mode='same', activation='relu',
                                               name='inception_4a/pool_proj', W_regularizer=l2(0.0002))(
            inception_4a_pool)

        inception_4a_output = merge([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj],
                                    mode='concat', concat_axis=1, name='inception_4a/output')

        loss1_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss1/ave_pool')(inception_4a_output)

        loss1_conv = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='loss1/conv',
                                   W_regularizer=l2(0.0002))(loss1_ave_pool)

        loss1_flat = Flatten()(loss1_conv)

        loss1_fc = Dense(1024, activation='relu', name='loss1/fc', W_regularizer=l2(0.0002))(loss1_flat)

        loss1_drop_fc = Dropout(0.7)(loss1_fc)

        loss1_classifier = Dense(1000, name='loss1/classifier', W_regularizer=l2(0.0002))(loss1_drop_fc)

        loss1_classifier_act = Activation('softmax')(loss1_classifier)

        inception_4b_1x1 = Convolution2D(160, 1, 1, border_mode='same', activation='relu', name='inception_4b/1x1',
                                         W_regularizer=l2(0.0002))(inception_4a_output)

        inception_4b_3x3_reduce = Convolution2D(112, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4b/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_4a_output)

        inception_4b_3x3 = Convolution2D(224, 3, 3, border_mode='same', activation='relu', name='inception_4b/3x3',
                                         W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)

        inception_4b_5x5_reduce = Convolution2D(24, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4b/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_4a_output)

        inception_4b_5x5 = Convolution2D(64, 5, 5, border_mode='same', activation='relu', name='inception_4b/5x5',
                                         W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)

        inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_4b/pool')(
            inception_4a_output)

        inception_4b_pool_proj = Convolution2D(64, 1, 1, border_mode='same', activation='relu',
                                               name='inception_4b/pool_proj', W_regularizer=l2(0.0002))(
            inception_4b_pool)

        inception_4b_output = merge([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj],
                                    mode='concat', concat_axis=1, name='inception_4b_output')

        inception_4c_1x1 = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='inception_4c/1x1',
                                         W_regularizer=l2(0.0002))(inception_4b_output)

        inception_4c_3x3_reduce = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4c/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_4b_output)

        inception_4c_3x3 = Convolution2D(256, 3, 3, border_mode='same', activation='relu', name='inception_4c/3x3',
                                         W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)

        inception_4c_5x5_reduce = Convolution2D(24, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4c/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_4b_output)

        inception_4c_5x5 = Convolution2D(64, 5, 5, border_mode='same', activation='relu', name='inception_4c/5x5',
                                         W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)

        inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_4c/pool')(
            inception_4b_output)

        inception_4c_pool_proj = Convolution2D(64, 1, 1, border_mode='same', activation='relu',
                                               name='inception_4c/pool_proj', W_regularizer=l2(0.0002))(
            inception_4c_pool)

        inception_4c_output = merge([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj],
                                    mode='concat', concat_axis=1, name='inception_4c/output')

        inception_4d_1x1 = Convolution2D(112, 1, 1, border_mode='same', activation='relu', name='inception_4d/1x1',
                                         W_regularizer=l2(0.0002))(inception_4c_output)

        inception_4d_3x3_reduce = Convolution2D(144, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4d/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_4c_output)

        inception_4d_3x3 = Convolution2D(288, 3, 3, border_mode='same', activation='relu', name='inception_4d/3x3',
                                         W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)

        inception_4d_5x5_reduce = Convolution2D(32, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4d/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_4c_output)

        inception_4d_5x5 = Convolution2D(64, 5, 5, border_mode='same', activation='relu', name='inception_4d/5x5',
                                         W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)

        inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_4d/pool')(
            inception_4c_output)

        inception_4d_pool_proj = Convolution2D(64, 1, 1, border_mode='same', activation='relu',
                                               name='inception_4d/pool_proj', W_regularizer=l2(0.0002))(
            inception_4d_pool)

        inception_4d_output = merge([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj],
                                    mode='concat', concat_axis=1, name='inception_4d/output')

        loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(inception_4d_output)

        loss2_conv = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='loss2/conv',
                                   W_regularizer=l2(0.0002))(loss2_ave_pool)

        loss2_flat = Flatten()(loss2_conv)

        loss2_fc = Dense(1024, activation='relu', name='loss2/fc', W_regularizer=l2(0.0002))(loss2_flat)

        loss2_drop_fc = Dropout(0.7)(loss2_fc)

        loss2_classifier = Dense(1000, name='loss2/classifier', W_regularizer=l2(0.0002))(loss2_drop_fc)

        loss2_classifier_act = Activation('softmax')(loss2_classifier)

        inception_4e_1x1 = Convolution2D(256, 1, 1, border_mode='same', activation='relu', name='inception_4e/1x1',
                                         W_regularizer=l2(0.0002))(inception_4d_output)

        inception_4e_3x3_reduce = Convolution2D(160, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4e/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_4d_output)

        inception_4e_3x3 = Convolution2D(320, 3, 3, border_mode='same', activation='relu', name='inception_4e/3x3',
                                         W_regularizer=l2(0.0002))(inception_4e_3x3_reduce)

        inception_4e_5x5_reduce = Convolution2D(32, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4e/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_4d_output)

        inception_4e_5x5 = Convolution2D(128, 5, 5, border_mode='same', activation='relu', name='inception_4e/5x5',
                                         W_regularizer=l2(0.0002))(inception_4e_5x5_reduce)

        inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_4e/pool')(
            inception_4d_output)

        inception_4e_pool_proj = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                               name='inception_4e/pool_proj', W_regularizer=l2(0.0002))(
            inception_4e_pool)

        inception_4e_output = merge([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj],
                                    mode='concat', concat_axis=1, name='inception_4e/output')

        inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)

        pool4_helper = PoolHelper()(inception_4e_output_zero_pad)

        pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool4/3x3_s2')(
            pool4_helper)

        inception_5a_1x1 = Convolution2D(256, 1, 1, border_mode='same', activation='relu', name='inception_5a/1x1',
                                         W_regularizer=l2(0.0002))(pool4_3x3_s2)

        inception_5a_3x3_reduce = Convolution2D(160, 1, 1, border_mode='same', activation='relu',
                                                name='inception_5a/3x3_reduce', W_regularizer=l2(0.0002))(pool4_3x3_s2)

        inception_5a_3x3 = Convolution2D(320, 3, 3, border_mode='same', activation='relu', name='inception_5a/3x3',
                                         W_regularizer=l2(0.0002))(inception_5a_3x3_reduce)

        inception_5a_5x5_reduce = Convolution2D(32, 1, 1, border_mode='same', activation='relu',
                                                name='inception_5a/5x5_reduce', W_regularizer=l2(0.0002))(pool4_3x3_s2)

        inception_5a_5x5 = Convolution2D(128, 5, 5, border_mode='same', activation='relu', name='inception_5a/5x5',
                                         W_regularizer=l2(0.0002))(inception_5a_5x5_reduce)

        inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_5a/pool')(
            pool4_3x3_s2)

        inception_5a_pool_proj = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                               name='inception_5a/pool_proj', W_regularizer=l2(0.0002))(
            inception_5a_pool)

        inception_5a_output = merge([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj],
                                    mode='concat', concat_axis=1, name='inception_5a/output')

        inception_5b_1x1 = Convolution2D(384, 1, 1, border_mode='same', activation='relu', name='inception_5b/1x1',
                                         W_regularizer=l2(0.0002))(inception_5a_output)

        inception_5b_3x3_reduce = Convolution2D(192, 1, 1, border_mode='same', activation='relu',
                                                name='inception_5b/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_5a_output)

        inception_5b_3x3 = Convolution2D(384, 3, 3, border_mode='same', activation='relu', name='inception_5b/3x3',
                                         W_regularizer=l2(0.0002))(inception_5b_3x3_reduce)

        inception_5b_5x5_reduce = Convolution2D(48, 1, 1, border_mode='same', activation='relu',
                                                name='inception_5b/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_5a_output)

        inception_5b_5x5 = Convolution2D(128, 5, 5, border_mode='same', activation='relu', name='inception_5b/5x5',
                                         W_regularizer=l2(0.0002))(inception_5b_5x5_reduce)

        inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_5b/pool')(
            inception_5a_output)

        inception_5b_pool_proj = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                               name='inception_5b/pool_proj', W_regularizer=l2(0.0002))(
            inception_5b_pool)

        inception_5b_output = merge([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj],
                                    mode='concat', concat_axis=1, name='inception_5b/output')

        pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='pool5/7x7_s2')(inception_5b_output)

        loss3_flat = Flatten()(pool5_7x7_s1)

        pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)

        loss3_classifier = Dense(1000, name='loss3/classifier', W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)

        loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

        googlenet = Model(input=input, output=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

        if weights_path:
            googlenet.load_weights(weights_path)

        self.google_net = googlenet
        return googlenet

    def predict_googlenet(self, x):
        preds = self.google_net.predict(x)
        return [np.argmax(preds[0]), np.argmax(preds[1]), np.argmax(preds[2])]

    def test_googlenet(self, input_network):
        model = self.create_googlenet(input_network)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        if not self.data_generator:
            raise Exception("No data generator")
        data_generator = self.data_generator
        print("Test Data")
        signal_signal = 0
        signal_background = 0
        background_signal = 0
        background_background = 0
        for i, row in enumerate(data_generator._signal):
            if i >= 2500:
                break
            x = data_generator._convert_row(row)
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)
            print([np.argmax(preds[0]), np.argmax(preds[1]), np.argmax(preds[2])])
            result = np.argmax(preds[0])
            if result == 0:
                print("%s: s/s" % i)
                signal_signal += 1
            else:
                print("%s: s/b" % i)
                signal_background += 1
        for i, row in enumerate(data_generator._background):
            if i >= 2500:
                break
            x = data_generator._convert_row(row)
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)
            print([np.argmax(preds[0]), np.argmax(preds[1]), np.argmax(preds[2])])
            result = np.argmax(preds[0])
            if result == 0:
                print("%s: b/s" % i)
                background_signal += 1
            else:
                print("%s: b/b" % i)
                background_background += 1
        print("\tSignal\tBackground")
        print("Signal\t%s\t%s" % (signal_signal, signal_background))
        print("Background\t%s\t%s" % (background_signal, background_background))

    def train_googlenet(self, save_path, recovery=None):
        if recovery:
            model = self.create_googlenet(recovery)
        else:
            model = self.create_googlenet()
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        if not self.data_generator:
            raise Exception("No data generator")
        data = self.data_generator
        print("Start Train.")
        for i in range(0, 10000):
            if self.exit_signal:
                break
            print("=" * 64)
            print("Loop %s" % i)
            model.fit_generator(generator=data.train_generator(), samples_per_epoch=data.get_train_size(), nb_epoch=1,
                                validation_data=data.test_generator(), nb_val_samples=data.get_test_size(), verbose=1)
            score = model.evaluate_generator(generator=data.test_generator(), val_samples=data.get_test_size())
            print(score)
            # print some predict:
            for i in range(100):
                row_x, row_y = data.get_some_test(1)
                predict = self.predict_googlenet(row_x)
                print('Except', [np.argmax(row_y), np.argmax(row_y), np.argmax(row_y)])
                print('Answer', predict)
                print('---')
        model.save_weights(save_path)


t = Train()


def signal_handler(signum, frame):
    print("Try to save train data. It may take a long time")
    t.exit_signal = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", choices=["train", "test"])
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-s", "--save", required=True)
    parser.add_argument("-m", "--max-energy", type=float)
    parser.add_argument("-r", "--recovery")
    args = parser.parse_args()
    t.set_data(args.input)
    if args.max_energy:
        t.data_generator.set_max_energy(args.max_energy)
    if args.type == "train":
        signal.signal(signal.SIGINT, signal_handler)
        if args.recovery:
            t.train_googlenet(args.save, args.recovery)
        else:
            t.train_googlenet(args.save)
    else:
        t.test_googlenet(args.save)
