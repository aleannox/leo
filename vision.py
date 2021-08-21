import io
import datetime
import pathlib
import time

from d2go.model_zoo import model_zoo
from d2go.utils.demo_predictor import DemoPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import torch


VISION_MEMORY_PATH = pathlib.Path(__file__).parent / 'memory' / 'vision'


class PersonDetector:
    def __init__(self, config):
        self.recorder = RealSenseImageRecorder()
        model = model_zoo.get(config['detectron_model'], trained=True)
        self.predictor = DemoPredictor(
            model,
            min_size_test=config['detectron_min_size'],
            max_size_test=config['detectron_max_size'],
        )
        self.camera_last_seen = 0
        self.confidence_threshold = config['detectron_confidence_threshold']
        self.person_class = config['detectron_person_class']
        
    def get_largest_person_relative_x(self):
        _, outputs = self._run_prediction()
        if outputs is not None:
            return self.extract_largest_person_relative_x(outputs)

    def _run_prediction(self):
        image = self.recorder.record_image()
        if image is not None:
            self.camera_last_seen = time.time()
            logger.info("Detecting persons.")
            outputs = self.predictor(image)
            PersonDetector.save_detection(image, outputs)
            return image, outputs
        else:
            return None, None
    
    def extract_largest_person_relative_x(self, outputs):
        """
        0 if person is in center
        0.5 if at right edge
        -0.5 if at left edge

        Discard detections with confidence < confidence_threshold.
        """
        class_check = outputs['instances'].pred_classes == self.person_class
        confidence_check = (
            outputs['instances'].scores >= self.confidence_threshold
        )
        valid_person_check = torch.bitwise_and(class_check, confidence_check)
        n_valid_persons = valid_person_check.sum()
        if not n_valid_persons:
            logger.info("No valid person detected.")
            if class_check.sum():
                max_confidence = outputs['instances'].scores[class_check].max()
                logger.info(
                    f"But I found {class_check.sum()} invalid persons with "
                    f"max confidence {max_confidence:.02f}."
                )
            return
        logger.info(f"I found {n_valid_persons} valid person(s).")
        largest_person_x_width = 0
        relative_x = None
        image_width = outputs["instances"].image_size[1]
        valid_person_boxes = outputs["instances"].pred_boxes[valid_person_check]
        for box_ in valid_person_boxes:
            box = box_.numpy()
            x_width = box[2] - box[0]
            if x_width > largest_person_x_width:
                largest_person_x_width = x_width
                relative_x = (box[0] + x_width / 2) / image_width - 0.5
        logger.info(f"The largest person is at relative_x = {relative_x:.02f}.")
        return relative_x

    @staticmethod
    def save_detection(image, outputs):
        v = Visualizer(
            image[:, :, ::-1], MetadataCatalog.get("coco_2017_train")
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure()
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
        )
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        filename = VISION_MEMORY_PATH / (
            datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.png'
        )
        logger.info(f"Saving detection at {filename}")
        plt.savefig(str(filename), bbox_inches='tight', pad_inches=0)


class RealSenseImageRecorder:
    def __init__(self):
        self._init_pipeline()
        
    def _init_pipeline(self):
        logger.info("Initializing recording pipeline.")
        self.pipeline = None
        try:
            self.pipeline = rs.pipeline()
            profile = self.pipeline.start()
            color_sensor = profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.gain, 99)
        except RuntimeError as e:
            logger.error(f"Could not initialize recording pipeline: {e}")
        
    def record_image(self, depth_share=0):
        logger.info("Recording image.")
        if self.pipeline is None:
            self._init_pipeline()
            if self.pipeline is None:
                return
        try:
            frames = self.pipeline.wait_for_frames()
            image_color = np.asarray(frames.get_color_frame().data).copy()
            if depth_share:
                image_depth = np.asarray(frames.get_depth_frame().data).copy()
            if depth_share:
                return (
                    image_color * (1 - depth_share)
                    + image_depth * depth_share
                ).round().astype('uint8')
            else:
                return image_color  # already 'uint8'
        except RuntimeError as e:
            logger.error(f"Could not record image: {e}")
            # Reset pipeline, we will try to restart it on next call.
            self.pipeline = None
