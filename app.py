# import time
# import random
# import numpy as np
# from absl import app, flags, logging
# from absl.flags import FLAGS
# import cv2
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from yolov3_tf2.models import YoloV3, YoloV3Tiny
# from yolov3_tf2.dataset import transform_images
# from yolov3_tf2.utils import draw_outputs, convert_boxes

# from deep_sort import preprocessing
# from deep_sort import nn_matching
# from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker
# from tools import generate_detections as gdet

# from PIL import Image

# flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
# flags.DEFINE_string('weights', './weights/yolov3.tf', 'path to weights file')
# flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_string('video', './data/video/test.mp4', 'path to video file or number for webcam)')
# flags.DEFINE_string('output', None, 'path to output video')
# flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
# flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


# def main(_argv):
#     # Definition of the parameters
#     max_cosine_distance = 0.5
#     nn_budget = None
#     nms_max_overlap = 1.0
    
#     # Initialize deep sort
#     model_filename = 'model_data/mars-small128.pb'
#     encoder = gdet.create_box_encoder(model_filename, batch_size=1)
#     metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#     tracker = Tracker(metric)

#     # Define marking area (Example: center area of the frame)
#     left_boundary = 100
#     right_boundary = 500
#     top_boundary = 100
#     bottom_boundary = 400

#     left_objects = []  # List of objects within the marking area

#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     if len(physical_devices) > 0:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)

#     if FLAGS.tiny:
#         yolo = YoloV3Tiny(classes=FLAGS.num_classes)
#     else:
#         yolo = YoloV3(classes=FLAGS.num_classes)

#     yolo.load_weights(FLAGS.weights)
#     logging.info('weights loaded')

#     class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
#     logging.info('classes loaded')

#     try:
#         vid = cv2.VideoCapture(int(FLAGS.video))
#     except:
#         vid = cv2.VideoCapture(FLAGS.video)

#     out = None

#     if FLAGS.output:
#         # By default VideoCapture returns float instead of int
#         width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(vid.get(cv2.CAP_PROP_FPS))
#         codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
#         out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
#         list_file = open('detection.txt', 'w')
#         frame_index = -1 
    
#     fps = 0.0
#     count = 0

#     while True:
#         _, img = vid.read()

#         if img is None:
#             logging.warning("Empty Frame")
#             time.sleep(0.1)
#             count += 1
#             if count < 3:
#                 continue
#             else:
#                 break

#         img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_in = tf.expand_dims(img_in, 0)
#         img_in = transform_images(img_in, FLAGS.size)

#         t1 = time.time()
#         boxes, scores, classes, nums = yolo.predict(img_in)
#         classes = classes[0]
#         names = []
#         for i in range(len(classes)):
#             names.append(class_names[int(classes[i])])
#         names = np.array(names)
#         converted_boxes = convert_boxes(img, boxes[0])
#         features = encoder(img, converted_boxes)
#         detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]

#         # Run non-maxima suppression
#         boxs = np.array([d.tlwh for d in detections])
#         scores = np.array([d.confidence for d in detections])
#         classes = np.array([d.class_name for d in detections])
#         indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
#         detections = [detections[i] for i in indices]

#         # Initialize color map
#         cmap = plt.get_cmap('tab20b')
#         colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

#         for det in detections:
#             bbox = det.to_tlbr()

#             # Check if the box is within the marking area
#             if left_boundary < bbox[0] < right_boundary and top_boundary < bbox[1] < bottom_boundary:
#                 # Object is within the marking area
#                 left_objects.append({'bbox': bbox, 'timestamp': time.time()})
#                 cv2.rectangle(img, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
#                 current_time = time.time()

#                 # If an object is in the list for more than 5 seconds, consider it left behind
#                 for obj in left_objects:
#                     if current_time - obj['timestamp'] > 5:
#                         # Action for left-behind object
#                         # For example, you can mark or send a notification
#                         cv2.putText(img, "Barang Ditinggalkan", (int(obj['bbox'][0]), int(obj['bbox'][1] - 10)), 0, 0.75, (255, 0, 0), 2)

#         cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
#         cv2.imshow('output', img)
        
#         if FLAGS.output:
#             out.write(img)
#             frame_index = frame_index + 1
#             list_file.write(str(frame_index) + ' ')
#             if len(converted_boxes) != 0:
#                 for i in range(0, len(converted_boxes)):
#                     list_file.write(str(converted_boxes[i][0]) + ' ' + str(converted_boxes[i][1]) + ' ' + str(converted_boxes[i][2]) + ' ' + str(converted_boxes[i][3]) + ' ')
#             list_file.write('\n')

#         # Press 'q' to quit
#         if cv2.waitKey(1) == ord('q'):
#             break

#     vid.release()

#     if FLAGS.output:
#         out.release()
#         list_file.close()

#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     try:
#         app.run(main)
#     except SystemExit:
#         pass

import time
import random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from PIL import Image

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Define marking area (Example: center area of the frame)
    left_boundary = 100
    right_boundary = 500
    top_boundary = 100
    bottom_boundary = 400

    left_objects = {} # Dictionary of objects within the marking area

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # By default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
    
    fps = 0.0
    count = 0

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count += 1
            if count < 3:
                continue
            else:
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        # with open("./data/labels/coco.names", "r") as f:
        #     classes = f.read().strip().split('\n')\
            
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]

        # Run non-maxima suppression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # Check if the box is within the marking area
            if left_boundary < bbox[0] < right_boundary and top_boundary < bbox[1] < bottom_boundary:
                if track.track_id not in left_objects:
                    if class_name == "chair":
                        # Object is within the marking area and is a backpack
                        left_objects[track.track_id] = {'bbox': bbox, 'timestamp': time.time(), 'lastSeen': time.time()}
                        # left_objects.append({'bbox': bbox, 'timestamp': time.time()})
                    if class_name == "backpack":
                        # Object is within the marking area and is a backpack
                        left_objects[track.track_id] = {'bbox': bbox, 'timestamp': time.time(), 'lastSeen': time.time()}
                        # left_objects.append({'bbox': bbox, 'timestamp': time.time()})
                else:
                    # Get object from the dictionary
                    obj = left_objects[track.track_id]
                    # Calculate centroid of the current object
                    currCentroid = ((bbox[0] + bbox[2]) / 2,(bbox[1] + bbox[3]) / 2)
                    # Calculate centroid of the last seen object
                    objCentroid = ((obj['bbox'][0] + obj['bbox'][2]) / 2,(obj['bbox'][1] + obj['bbox'][3]) / 2)
                    # Calculate distance between current and last seen
                    distance = abs(currCentroid[0]-objCentroid[0]) + abs(currCentroid[1]-objCentroid[1])
                    # If the distance more than 100px, update position and timestamp
                    if distance > 100:
                        left_objects[track.track_id] = {'bbox': bbox, 'timestamp': time.time()}
                    # Update the object last seen
                    left_objects[track.track_id]['lastSeen'] = time.time()

        # Check for backpacks that have stopped moving
        current_time = time.time()
        for key in left_objects.keys():
            if current_time - left_objects[key]['timestamp'] > 5:
                # Action for left-behind backpack
                # For example, you can mark or send a notification
                cv2.putText(img, "Barang Ditinggalkan", (int(left_objects[key]['bbox'][0]), int(left_objects[key]['bbox'][1] - 10)), 0, 0.75, (255, 0, 0), 2)

                # Draw a bounding box around the left-behind object
                cv2.rectangle(img, (int(left_objects[key]['bbox'][0]), int(left_objects[key]['bbox'][1])), (int(left_objects[key]['bbox'][2]), int(left_objects[key]['bbox'][3])), (0, 0, 255), 2)

            # if current_time - left_objects[key]['lastSeen'] > 5:
            #     left_objects.pop(key, None)

        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)
        
        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(converted_boxes) != 0:
                for i in range(0, len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' ' + str(converted_boxes[i][1]) + ' ' + str(converted_boxes[i][2]) + ' ' + str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    if FLAGS.output:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass