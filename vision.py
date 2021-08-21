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


VISION_MEMORY_PATH = pathlib.Path(__file__).parent / 'memory' / 'vision'


class PersonDetector:
    def __init__(self):
        self.recorder = RealSenseImageRecorder()
        model = model_zoo.get(
            'faster_rcnn_fbnetv3a_dsmask_C4.yaml', trained=True
        )
        self.predictor = DemoPredictor(model)
        self.camera_last_seen = 0
        
    def get_largest_person_relative_x(self):
        _, outputs = self._run_prediction()
        if outputs is not None:
            return PersonDetector.extract_largest_person_relative_x(outputs)

    def _run_prediction(self):
        image = self.recorder.record_image()
        if image is not None:
            self.camera_last_seen = time.time()
            logger.info("Detecting persons.")
            outputs = self.predictor(image)
            logger.info("Prediction classes")
            logger.info(outputs["instances"].pred_classes)
            logger.info("Prediction boxes")
            logger.info(outputs["instances"].pred_boxes)
            PersonDetector.save_detection(image, outputs)
            return image, outputs
        else:
            return None, None
        
    PERSON_CLASS = 0  # COCO 2017 used in model
    
    @staticmethod
    def extract_largest_person_relative_x(outputs):
        """
        0 if person is in center
        0.5 if at right edge
        -0.5 if at left edge
        """
        if PersonDetector.PERSON_CLASS not in outputs["instances"].pred_classes:
            logger.info("No person detected.")
            return
        logger.info(f"I found {(outputs['instances'].pred_classes == 0).sum()} person(s).")
        largest_person_x_width = 0
        relative_x = None
        image_width = outputs["instances"].image_size[1]
        for index, class_ in enumerate(outputs["instances"].pred_classes):
            if class_ == PersonDetector.PERSON_CLASS:
                box = outputs["instances"].pred_boxes[index].tensor[0].numpy()
                x_width = box[2] - box[0]
                if x_width > largest_person_x_width:
                    largest_person_x_width = x_width
                    relative_x = (box[0] + x_width / 2) / image_width - 0.5
        logger.info(f"The largest person is at relative_x = {relative_x}.")
        return relative_x

    @staticmethod
    def save_detection(image, outputs):
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("coco_2017_train"))
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure()
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
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
            image_color = np.asarray(frames.get_color_frame().data)
            if depth_share:
                image_depth = np.asarray(frames.get_depth_frame().data)
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
