# prepare dataset and train
import time
from time import sleep

import skimage
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure

import model as modellib
import config
import utils
import visualize

import sys
import os
import warnings
from random import seed
from random import randint

import cv2
import numpy as np
import threading
import datetime
import matplotlib.pyplot as plt
from IPython import display

warnings.filterwarnings("ignore")

frames = []
results = []
frame = None


# class that defines and loads the weapon dataset
class WeaponDataset(utils.Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "weapon")
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[7:-5]
            # skip all images after 600 if we are building the train set
            if is_train and int(image_id) < 600:
                continue
            # skip all images before 600 if we are building the test/val set
            if not is_train and int(image_id) >= 600:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + filename[:-4] + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('weapon'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# define a configuration for the model
class WeaponConfig(config.Config):
    # define the name of the configuration
    NAME = "weapon_cfg"
    # number of classes (background + weapon)
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 500

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class PredictionConfig(config.Config):
    # define the name of the configuration
    NAME = "weapon_cfg"
    # number of classes (background + weapon)
    NUM_CLASSES = 1 + 1

    # BATCH_SIZE = 2


def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id,
                                                                                  use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = modellib.mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        try:
            # calculate statistics, including AP
            AP, _, _, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"],
                                           r['masks'])
            # store
            APs.append(AP)
            print("good image id : %d" % image_id)
            print("AP : %.3f" % AP)
        except:
            print("bad image id : %d" % image_id)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP


# prediction and visualize
def testPredict(model, dataset):
    for _ in range(10):
        i = randint(0, 150)
        print('--------------------random number : %d' % i)
        image = dataset.load_image(i)
        results = model.detect([image], verbose=1)
        # Visualize results
        r = results[0]
        class_names = ['BG', 'weapon']
        print(r['scores'])
        list = r['scores']
        is_display = True
        for i in range(len(list)):
            if list[i] < 0.9:
                is_display = False
                continue

        if is_display:
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'])


def predict():
    # prepare config
    cfg = PredictionConfig()
    cfg.BATCH_SIZE = 1
    # define the model
    model = modellib.MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model.load_weights('mask_rcnn_weapon_cfg_0009.h5', by_name=True)

    for _ in range(10):
        ROOT_DIR = os.getcwd()
        IMAGE_DIR = os.path.join(ROOT_DIR, "armedperson")
        file_names = next(os.walk(IMAGE_DIR))[2]
        i = randint(0, 332)
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[i]))
        results = model.detect([image], verbose=1)
        # Visualize results
        r = results[0]
        class_names = ['BG', 'weapon']
        print(r['scores'])
        list = r['scores']
        is_display = True
        # for i in range(len(list)):
        #     if list[i] < 0.9:
        #         is_display = False
        #         continue

        if is_display:
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'])


# plot a number of photo11s with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=1):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = modellib.mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        pyplot.subplot(n_images, 2, i * 2 + 1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        pyplot.subplot(n_images, 2, i * 2 + 2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
    # show the figure
    pyplot.show()


def train(train_set, test_set):
    # prepare config
    cfg = WeaponConfig()
    cfg.display()
    # define the model
    model = modellib.MaskRCNN(mode='training', model_dir='./', config=cfg)
    # load weights (mscoco) and exclude the output layers
    model.load_weights('mask_rcnn_coco.h5', by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    model.train(train_set, test_set, learning_rate=cfg.LEARNING_RATE, epochs=5, layers='heads')


def evaluate(train_set, test_set):
    # prepare config
    cfg = PredictionConfig()
    # define the model
    model = modellib.MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model.load_weights('mask_rcnn_weapon_cfg_0009.h5', by_name=True)
    # evaluate model on training dataset
    train_mAP = evaluate_model(train_set, model, cfg)
    print("Train mAP: %.3f" % train_mAP)
    # evaluate model on test dataset
    test_mAP = evaluate_model(test_set, model, cfg)
    print("Test mAP: %.3f" % test_mAP)


def plot(train_set, test_set):
    # prepare config
    cfg = PredictionConfig()
    # define the model
    model = modellib.MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model.load_weights('mask_rcnn_weapon_cfg_0009.h5', by_name=True)
    figure(num=None, figsize=(32, 24), dpi=80, facecolor='w', edgecolor='k')
    # plot predictions for train dataset
    plot_actual_vs_predicted(train_set, model, cfg)
    figure(num=None, figsize=(32, 24), dpi=80, facecolor='w', edgecolor='k')
    # plot predictions for test dataset
    plot_actual_vs_predicted(test_set, model, cfg)  # prepare train set


def display(test_set):
    # prepare config
    cfg = PredictionConfig()
    # define the model
    cfg.BATCH_SIZE = 1
    model = modellib.MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model.load_weights('mask_rcnn_weapon_cfg_0009.h5', by_name=True)
    testPredict(model, test_set)


def captureWebcam():
    global frames
    global results
    global frame
    # cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    rval = False
    if vc.isOpened():  # try to get the first frame
        rval = True
    VIDEO_SAVE_DIR = './'
    class_names = ['BG', 'weapon']
    frame_count = 0
    keras_thread = MyThread()
    keras_thread.start()
    while rval:
        rval, frame = vc.read()
        if not rval:
            break
        cv2.imshow("preview", frame)
        frame_count += 1
        frames.append(frame)
        if len(results) > 0:
            # time.sleep(1)
            r = results[0]
            print(r['scores'])
            show_frame = False
            for score in r['scores']:
                if score > 0.9:
                    show_frame = True

            if show_frame:
                frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
                cv2.imshow("preview1", frame)
                time.sleep(5)
            results = []

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            frames = []
            break
    cv2.destroyWindow("preview")


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        if score > 0.9:
            image = apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            image = cv2.putText(
                image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
            )

    return image


class MyThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global frames
        global results
        global frame
        # prepare config
        cfg = PredictionConfig()
        # define the model
        mrcnn_model = modellib.MaskRCNN(mode='inference', model_dir='./', config=cfg)
        # load model weights
        mrcnn_model.load_weights('mask_rcnn_weapon_cfg_0009.h5', by_name=True)
        class_names = ['BG', 'weapon']
        while len(frames) > 0:
            if len(frames) >= cfg.BATCH_SIZE:
                start = datetime.datetime.now().replace(microsecond=0)
                results = mrcnn_model.detect(frames[-2:], verbose=0)
                end = datetime.datetime.now().replace(microsecond=0)
                print("detection time : %s" % (end - start))
                for i, item in enumerate(zip(frames, results)):
                    frame = item[0]


def captureVideo():
    # We use a K80 GPU with 24GB memory, which can fit 3 images.
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
    VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save")
    WEAPON_MODEL_PATH = "mask_rcnn_weapon_cfg_0009.h5"
    if not os.path.exists(WEAPON_MODEL_PATH):
        utils.download_trained_weights(WEAPON_MODEL_PATH)

    config = PredictionConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )
    model.load_weights(WEAPON_MODEL_PATH, by_name=True)
    class_names = ['BG', 'weapon']

    capture = cv2.VideoCapture('video/trailer.mkv')
    try:
        if not os.path.exists(VIDEO_SAVE_DIR):
            os.makedirs(VIDEO_SAVE_DIR)
    except OSError:
        print('Error: Creating directory of data')
    frames = []
    frame_count = 0
    # these 2 lines can be removed if you dont have a 1080p camera.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        # Bail out when the video file ends
        if not ret:
            break

        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)
        print('frame_count :{0}'.format(frame_count))
        if len(frames) == config.BATCH_SIZE:
            results = model.detect(frames, verbose=0)
            print('Predicted')
            for i, item in enumerate(zip(frames, results)):
                frame = item[0]
                r = item[1]
                frame = display_instances(
                    frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
                )
                if frame is not None:
                    name = '{0}.jpg'.format(frame_count + i - config.BATCH_SIZE)
                    name = os.path.join(VIDEO_SAVE_DIR, name)
                    cv2.imwrite(name, frame)
                    print('writing to file:{0}'.format(name))
            # Clear the frames array to start the next batch
            frames = []

    capture.release()


# captureVideo()

#captureWebcam()

predict()

# prepare test/val set
# test_set = WeaponDataset()
# test_set.load_dataset('weapon', is_train=False)
# test_set.prepare()
# print('Test: %d' % len(test_set.image_ids))
# display(test_set)

# prepare train set
# train_set = WeaponDataset()
# train_set.load_dataset('weapon', is_train=True)
# train_set.prepare()
# print('Train: %d' % len(train_set.image_ids))

# train(train_set, test_set)
# evaluate(train_set, test_set)
# plot(train_set, test_set)
