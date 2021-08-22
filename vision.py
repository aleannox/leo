import datetime
import pathlib
import time

import d2go.export.api
import d2go.export.d2_meta_arch
import d2go.model_zoo
import d2go.utils.demo_predictor
import d2go.utils.testing.data_loader_helper
import detectron2.utils.visualizer
import detectron2.data
from loguru import logger
import matplotlib.pyplot as plt
import mobile_cv.predictor.api
import numpy as np
import pyrealsense2 as rs
import torch


VISION_MEMORY_PATH = pathlib.Path(__file__).parent / 'memory' / 'vision'

QUANTIZATION_TYPE = 'torchscript_int8@tracing'
QUANTIZED_MODELS_PATH = pathlib.Path(__file__).parent / 'quantized_models'


class PersonDetector:
    def __init__(self, config):
        self.recorder = RealSenseImageRecorder()
        if config['detectron_use_quantized_model']:
            model = PersonDetector._get_quantized_model(
                config['detectron_model']
            )
        else:
            model = d2go.model_zoo.model_zoo.get(
                config['detectron_model'], trained=True
            )
        self.predictor = d2go.utils.demo_predictor.DemoPredictor(
            model,
            min_size_test=config['detectron_min_size'],
            max_size_test=config['detectron_max_size'],
        )
        self.camera_last_seen = 0
        self.confidence_threshold = config['detectron_confidence_threshold']
        self.person_class = config['detectron_person_class']
        self.save_image_interval = config['detectron_save_image_interval']
        self.image_last_saved = 0

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
            if time.time() - self.image_last_saved > self.save_image_interval:
                self.image_last_saved = time.time()
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
        v = detectron2.utils.visualizer.Visualizer(
            image[:, :, ::-1],
            detectron2.data.MetadataCatalog.get("coco_2017_train"),
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure()
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
        )
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        filename = VISION_MEMORY_PATH / (
            datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.png'
        )
        logger.info(f"Saving detection at {filename}")
        plt.savefig(str(filename), bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def _get_quantized_model(model_name):
        quantized_path = PersonDetector._get_quantized_model_path(model_name)
        if not quantized_path.exists():
            logger.info("Creating quantized model because it doesn't exist.")
            PersonDetector._create_quantized_model(model_name)
        logger.info(f"Loading quantized model from {quantized_path}.")
        torch.backends.quantized.engine = 'qnnpack'
        # ^ https://github.com/pytorch/pytorch/issues/29327#issue-518778762
        return mobile_cv.predictor.api.create_predictor(str(quantized_path))

    @staticmethod
    def _get_quantized_model_path(model_name):
        return QUANTIZED_MODELS_PATH / model_name / QUANTIZATION_TYPE

    @staticmethod
    def _create_quantized_model(model_name):
        # Adapted from
        # https://github.com/facebookresearch/d2go/blob/main/demo/d2go_beginner.ipynb
        d2go.export.d2_meta_arch.patch_d2_meta_arch()
        model = d2go.model_zoo.model_zoo.get(model_name, trained=True)
        model.cpu()
        config = d2go.model_zoo.model_zoo.get_config(model_name)
        config.QUANTIZATION.BACKEND = 'qnnpack'
        # ^ https://github.com/pytorch/android-demo-app/issues/104
        with d2go.utils.testing.data_loader_helper.create_fake_detection_data_loader(  # noqa
            224, 320, is_train=False
        ) as data_loader:
            quantized_path = d2go.export.api.convert_and_export_predictor(
                config,
                model,
                QUANTIZATION_TYPE,
                str(QUANTIZED_MODELS_PATH / model_name),
                data_loader,
            )
            assert (
                quantized_path
                == str(PersonDetector._get_quantized_model_path(model_name))
            )


class RealSenseImageRecorder:
    def __init__(self):
        self._init_pipeline()

    def _init_pipeline(self):
        logger.info("Initializing recording pipeline.")
        try:
            self.pipeline = rs.pipeline()
            profile = self.pipeline.start()
            color_sensor = profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.gain, 99)
        except RuntimeError as e:
            logger.error(f"Could not initialize recording pipeline: {e}")
            # Reset pipeline, we will try to restart it on next call.
            self.pipeline = None

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
