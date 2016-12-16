import numpy as np
import time
import json
import argparse
import signal
from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.engine.training import slice_X
from googlenet_custom_layers import PoolHelper, LRN


class train:
    def __init__(self):
        self.google_net = None
        self.exit_signal = False
        self.thread = None

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

    def predict(self):
        img = imresize(imread('cat.jpg', mode='RGB'), (224, 224)).astype(np.float32)
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Test pretrained model
        model = self.create_googlenet('googlenet_weights.h5')
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        print("Start Compile:", time.time())
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        print("Start Predict:", time.time())
        out = model.predict(img)  # note: the model has three outputs
        print("Finish Predict:", time.time())
        print(np.argmax(out[0]), np.argmax(out[1]), np.argmax(out[2]))

    def test(self, input_data, input_network):
        model = self.create_googlenet(input_network)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        print("Load Data.")
        f = open(input_data, "r")
        data = json.loads(f.read())
        signal = data['signal']
        background = data['background']
        print("Test Data")
        signal_signal = 0
        signal_background = 0
        background_signal = 0
        background_background = 0
        X = np.zeros((1, 3, 224, 224), dtype='float32')
        for row in signal:
            X[0] = self.convert_row(row)
            result = np.argmax(model.predict(X)[0])
            if result == 0:
                signal_signal += 1
            else:
                signal_background += 1
        for row in background:
            X[0] = self.convert_row(row)
            result = np.argmax(model.predict(X)[0])
            if result == 0:
                background_signal += 1
            else:
                background_background += 1
        print("\tSignal\tBackground")
        print("Signal\t%s\t%s" % (signal_signal, signal_background))
        print("Background\t%s\t%s" % (background_signal, background_background))

    def convert_row(self, input_data):
        row = np.zeros((3, 224, 224))
        cluster_xy_data = input_data[0]
        for pixel, energy in cluster_xy_data.items():
            location = pixel.split(":")
            location_x = int(location[0])
            location_y = int(location[1])
            location_x += 224 / 2
            location_y += 224 / 2
            if not (0 <= location_x < 224 and 0 <= location_y < 224):
                continue
            row[0, location_x, location_y] = energy
        cluster_zy_data = input_data[1]
        for pixel, energy in cluster_zy_data.items():
            location = pixel.split(":")
            location_z = int(location[0])
            location_y = int(location[1])
            location_z += 224 / 2
            location_y += 224 / 2
            if not (0 <= location_z < 224 and 0 <= location_y < 224):
                continue
            row[1, location_z, location_y] = energy
        return row

    def prepare_data(self, input_data):
        print("Load Data.")
        f = open(input_data, "r")
        data = json.loads(f.read())
        signal = data['signal']
        background = data['background']
        print("Total %s Event" % (len(signal) + len(background)))
        X = np.zeros((5000, 3, 224, 224), dtype='float32')
        Y = np.zeros((5000, 1000), dtype='float32')
        for i, s in enumerate(signal):
            if i >= 2500:
                break
            X[i] = self.convert_row(s)
            Y[i][0] = 1

        for i, b in enumerate(background):
            if i >= 2500:
                break
            X[i + 2500] = self.convert_row(b)
            Y[i + 2500][1] = 1

        print("Load Data Finished.")
        return X, Y

    def train(self, input_data, save_path):
        model = self.create_googlenet()
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        X, Y = self.prepare_data(input_data)
        # Random Data List
        indices = np.arange(len(Y))
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        # Split 10% as test data
        split_at = len(X) - len(X) / 10
        (X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
        (Y_train, Y_val) = (Y[:split_at], Y[split_at:])
        print("X shape: ", X_train.shape)
        print("Y shape: ", Y_train.shape)

        print("Start Train.")
        for i in range(0, 10000):
            if self.exit_signal:
                break
            print("=" * 64)
            print("Loop %s" % i)
            model.fit(X_train, [Y_train, Y_train, Y_train], batch_size=128, nb_epoch=1,
                      validation_data=(X_val, [Y_val, Y_val, Y_val]))

            # print some test:
            for i in range(10):
                ind = np.random.randint(0, len(X_val))
                rowX, rowy = X_val[np.array([ind])], Y_val[np.array([ind])]
                preds = model.predict(rowX, verbose=0)
                preds = [np.argmax(preds[0]), np.argmax(preds[1]), np.argmax(preds[2])]
                print('Except', [np.argmax(rowy), np.argmax(rowy), np.argmax(rowy)])
                print('Answer', preds)
                print('---')

        self.google_net.save(save_path)


t = train()


def signal_handler(signum, frame):
    print("Try to save train data. It may take a long time")
    t.exit_signal = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", choices=["train", "test"])
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-s", "--save", required=True)
    args = parser.parse_args()
    if args.type == "train":
        signal.signal(signal.SIGINT, signal_handler)
        t.train(args.input, args.save)
    else:
        t.test(args.input, args.save)
