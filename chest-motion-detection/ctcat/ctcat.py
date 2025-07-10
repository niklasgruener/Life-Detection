import json
import os
import glob
import time
import logging
from queue import Queue, Empty
from enum import IntEnum

import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api

from CTCAT_software_RGBDT.uvctypes import *

def add_text(img, text, loc, color, font=cv2.FONT_HERSHEY_SIMPLEX):
    textsize = cv2.getTextSize(text, font, 1, 1)[0]
    x = (loc[1] * 2 - textsize[0]) // 2
    y = (loc[0] * 2 + textsize[1]) // 2
    cv2.putText(img, text, (x, y), font, 1, color, 1)

class CTCAT_Orbbec:
    def __init__(self, device):
        self.dll_directories = {
            'linux-x64': 'OpenNI-Linux-x64-2.3.0.63/Redist/',
            'linux-arm': 'OpenNI-Linux-Arm-2.3.0.66/Redist',
            'linux-arm64': 'OpenNI-Linux-Arm64-2.3.0.63/Redist',
            'darwin-x64': 'AstraSDK-0.5.0-20160426T102621Z-darwin-x64/lib/Plugins/openni2',
            'windows-x64': 'OpenNi-Windows-x64-2.3.0.66/Redist/OpenNI2/Drivers'
        }[device]

        self.orbbec, self.depth_stream, self.color_stream = self.initialize_sensor()

    def __del__(self):
        openni2.unload()

    def initialize_sensor(self):
        openni2.initialize(dll_directories=os.path.join('CTCAT_software_RGBDT', self.dll_directories))
        if not openni2.is_initialized():
            msg = "Could not initialize openni!"
            logging.error(msg)
            raise RuntimeError(msg)

        try:
            OrbbecAstra = openni2.Device.open_any()
        except Exception as e:
            logging.error("Could not open Orbbec Astra!")
            raise

        # enable stream synchronization
        OrbbecAstra.set_depth_color_sync_enabled(True)

        # set up depth stream
        depthStream = OrbbecAstra.create_depth_stream()
        depthStream.set_video_mode(
            c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640,
                               resolutionY=480, fps=30))
        # set up rgb stream
        colorStream = OrbbecAstra.create_color_stream()
        colorStream.set_video_mode(
            c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640,
                               resolutionY=480,
                               fps=30))

        # start streams
        try:
            depthStream.start()
        except Exception as e:
            logging.error("Depth stream could not be started!")
            raise
        logging.info("Depth stream started!")
        try:
            colorStream.start()
        except Exception as e:
            logging.error("Color stream could not be started!")
            raise
        logging.info("Color stream started!")

        # set registration mode (depth->color or disabled)
        OrbbecAstra.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        # OrbbecAstra.set_image_registration_mode(openni2.IMAGE_REGISTRATION_OFF)

        # disable mirroring
        depthStream.set_mirroring_enabled(False)
        colorStream.set_mirroring_enabled(False)
        return OrbbecAstra, depthStream, colorStream

    def read(self):
        # read raw depth image
        depthFrame = self.depth_stream.read_frame()
        depthFrameData = depthFrame.get_buffer_as_uint16()
        depthImage16Bit = np.frombuffer(depthFrameData, dtype=np.uint16).copy() # copy to detach image from fixed address in memory
        depthImage16Bit.shape = (480, 640)

        # read color image
        colorFrame = self.color_stream.read_frame()
        colorFrameData = colorFrame.get_buffer_as_uint8()
        colorImage8Bit = np.frombuffer(colorFrameData, dtype=np.uint8).copy()
        colorImage8Bit.shape = (480, 640, 3)
        colorImage8Bit = cv2.cvtColor(colorImage8Bit, cv2.COLOR_BGR2RGB)

        return depthImage16Bit, colorImage8Bit

class CTCAT_Flir:
    def __init__(self, blocking_read=True):
        self.thermal_queue, self.pointers = self.initialize_sensor()
        self.blocking_read = blocking_read

    def __del__(self):
        if not hasattr(self, 'pointers'):
            return
        try:
            libuvc.uvc_stop_streaming(self.pointers[4])
            libuvc.uvc_unref_device(self.pointers[3])
            libuvc.uvc_exit(self.pointers[2])
        except AttributeError:
            logging.error("Error when garbage collection flir pointers")
            raise

    def initialize_sensor(self):
        # thermal image queue
        BUF_SIZE = 3
        q = Queue(maxsize=BUF_SIZE)

        def py_frame_callback(frame, userptr):
            array_pointer = cast(frame.contents.data,
                                 POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
            data = np.frombuffer(array_pointer.contents, dtype=np.dtype(np.uint16)).reshape(frame.contents.height,
                                                                                            frame.contents.width)
            if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
                return
            if not q.full():
                q.put(data)

        PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)
        ctx = POINTER(uvc_context)()
        dev = POINTER(uvc_device)()
        devh = POINTER(uvc_device_handle)()
        ctrl = uvc_stream_ctrl()

        res = libuvc.uvc_init(byref(ctx), 0)
        if res < 0:
            msg = "UVC initialization error!"
            logging.error(msg)
            raise RuntimeError(msg)

        try:
            res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
            if res < 0:
                msg = "Flir Lepton 3.5 not found!"
                logging.error(msg)
                raise RuntimeError(msg)
        except Exception as e:
            libuvc.uvc_exit(ctx)
            raise

        try:
            res = libuvc.uvc_open(dev, byref(devh))
            if res < 0:
                msg = "UVC open error!"
                logging.error(msg)
                raise RuntimeError(msg)

            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0:
                msg = "Thermal camera does not support 16bit raw!"
                logging.error(msg)
                raise RuntimeError(msg)

            libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16, frame_formats[0].wWidth,
                                                   frame_formats[0].wHeight,
                                                   int(1e7 / frame_formats[0].dwDefaultFrameInterval))
            res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
            if res < 0:
                msg = "Thermal stream could not be started: {0}".format(res)
                logging.error(msg)
                raise RuntimeError(msg)
            logging.info("Thermal stream started!")
        except Exception as e:
            libuvc.uvc_unref_device(dev)
            raise

        return q, (PTR_PY_FRAME_CALLBACK, uvc_frame, ctx, dev, devh)

    def read(self):
        try:
            thermal_frame = self.thermal_queue.get(block=self.blocking_read)
            thermalImage16Bit = np.frombuffer(thermal_frame, dtype=np.uint16)
            thermalImage16Bit.shape = (120, 160)
        except Empty:
            thermal_frame = np.zeros((120, 160), dtype=np.uint16)
            logging.warning('thermal frame was empty. returning blank frame instead.')

        return thermal_frame

class CTCAT_DataFormat(IntEnum):
    Raw = 1
    Resized = 2
    Registered = 3

class CTCAT_Source:
    def __init__(self, sensor_id, data_format=CTCAT_DataFormat.Raw, mirrored=False, colorize=False, fps=None):
        self._data_format = self.set_data_format(data_format)
        self._colorized = self.set_colorize(colorize)
        self._mirrored = mirrored
        self.params = self._load_params(sensor_id)

        self._fps = fps
        self._previous_time = time.time()

    def read(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        if self._fps is not None:
            current_time = time.time()
            sleep_time = 1 / self._fps - (current_time - self._previous_time)
            time.sleep(max(0, sleep_time))

        data = self.read()
        if data is None:
            return None
        else:
            depthFrame, colorFrame, thermalFrame = data

        if self._fps is not None:

            # while current_time < self._previous_time + 1 / self._fps:
            #     depthFrame, colorFrame, _thermalFrame = self.read()
            #     thermalFrame = _thermalFrame if thermalFrame is None else thermalFrame
            #     current_time = time.time()
            self._previous_time = current_time

        if thermalFrame is None:
            thermalFrame = np.zeros((120, 160), dtype=np.uint16)

        if self._data_format >= CTCAT_DataFormat.Resized:
            thermalFrame = cv2.resize(thermalFrame, (640, 480))

        if self._data_format >= CTCAT_DataFormat.Registered:
            depthFrame, colorFrame, thermalFrame = self._register(depthFrame, colorFrame, thermalFrame)

        if self._colorized:
            depthFrame, colorFrame, thermalFrame = self._colorize(depthFrame, colorFrame, thermalFrame)

        if self._mirrored:
            depthFrame, colorFrame, thermalFrame = self._mirror(depthFrame, colorFrame, thermalFrame)

        return depthFrame, colorFrame, thermalFrame

    def set_data_format(self, data_format):
        self._data_format = data_format
        return data_format

    def set_colorize(self, colorize):
        if self._data_format >= CTCAT_DataFormat.Resized:
            self._colorized = colorize
            return colorize
        if colorize:
            logging.warning('colorization not supported if data_format < CTCAT_DataFormat.Resized')

    def set_mirrored(self, mirrored_state):
        self._mirrored = mirrored_state

    def _load_params(self, sensor_id):
        # params_path = '/Users/thomas/data/mul/src/params/ctcat%s.json' % str(sensor_id).zfill(3)
        params_path = 'params/ctcat%s.json' % str(sensor_id).zfill(3)
        params = {key: np.array(value) for key, value in json.load(open(params_path)).items()}
        return params

    def _register(self, depth_frame, color_frame, thermal_frame):
        # undistort RGB image
        color_frame = cv2.undistort(color_frame, self.params['intrinsic_rgb'], self.params['distortion_rgb'])
        # register RGB to depth image
        color_frame = cv2.warpAffine(color_frame, self.params['registration_rgb'][:2], dsize=(640, 480))

        # undistort thermal image
        thermal_frame = cv2.undistort(thermal_frame, self.params['intrinsic_thermal'], self.params['distortion_thermal'])
        # register thermal to depth image
        background_temperature = 25
        background_value = (27315 + background_temperature * 100)
        # background_value = thermal_frame.mean()
        thermal_frame = cv2.warpAffine(thermal_frame, self.params['registration_thermal'][:2], dsize=(640, 480), borderValue=background_value)

        # depth_frame = cv2.warpAffine(depth_frame, self.params['registration_depth'][:2], dsize=(640, 480))
        return depth_frame, color_frame, thermal_frame

    def _colorize(self, depthFrame, colorFrame, thermalFrame):
        # normalize raw depth image for displaying
        depthFrame = cv2.normalize(depthFrame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # depthFrame = cv2.merge([(depthFrame * 255).astype(np.uint8)] * 3)
        depthFrame = (depthFrame * 255).astype(np.uint8)

        # normalize raw thermal image for displaying
        # thermalFrame = cv2.normalize(thermalFrame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        thermalFrame = (thermalFrame - 27315) / 100 # to celsius
        temp_bounds = (20, 40)
        thermalFrame = np.clip(0, 1, (thermalFrame - temp_bounds[0]) / (temp_bounds[1] - temp_bounds[0]))
        thermalFrame = (thermalFrame * 255).astype(np.uint8)
        return depthFrame, colorFrame, thermalFrame

    def _mirror(self, depthFrame, colorFrame, thermalFrame):
        colorFrame = cv2.flip(colorFrame, 1)
        depthFrame = cv2.flip(depthFrame, 1)
        thermalFrame = cv2.flip(thermalFrame, 1)
        return depthFrame, colorFrame, thermalFrame

class CTCAT_Sensor(CTCAT_Source):
    def __init__(self, sensor_id, data_format=CTCAT_DataFormat.Raw, mirrored=False, colorize=False, fps=None, blocking_thermal=True, device='darwin-x64'):
        self.orbbec = CTCAT_Orbbec(device=device)
        self.flir = CTCAT_Flir(blocking_read=blocking_thermal)

        super().__init__(sensor_id, data_format, mirrored, colorize, fps)

    def read(self):
        depthFrame, colorFrame = self.orbbec.read()
        thermalFrame = self.flir.read()
        return depthFrame, colorFrame, thermalFrame

    def set_data_format(self, data_format):
        self._data_format = data_format
        # self.orbbec.orbbec.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        self.orbbec.orbbec.set_image_registration_mode(openni2.IMAGE_REGISTRATION_OFF)
        # if data_format >= CTCAT_DataFormat.Registered:
        #     self.orbbec.orbbec.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        # else:
        #     self.orbbec.orbbec.set_image_registration_mode(openni2.IMAGE_REGISTRATION_OFF)
        return data_format

class CTCAT_Reader(CTCAT_Source):
    def __init__(self, directory, sensor_id, data_format=CTCAT_DataFormat.Raw, mirrored=False, colorize=False, fps=None):
        self._directory = directory
        self._depth_paths = sorted(glob.glob(os.path.join(directory, 'depth_*.png')))
        self._color_paths = sorted(glob.glob(os.path.join(directory, 'rgb_*.png')))
        self._thermal_paths = sorted(glob.glob(os.path.join(directory, 'thermal_*.png')))
        # self._depth_paths = sorted(glob.glob(os.path.join(directory, 'depth/*.png')))
        # self._color_paths = sorted(glob.glob(os.path.join(directory, 'rgb_raw/*.png')))
        # self._thermal_paths = sorted(glob.glob(os.path.join(directory, 'thermal_raw/*.png')))

        if not (len(self._depth_paths) == len(self._color_paths) == len(self._thermal_paths)):
            logging.warning('Warning: number of rgb, depth and thermal frames don\'t match.')

        self._n_frames = len(self._depth_paths)
        self._current_frame = 0

        super().__init__(sensor_id, data_format, mirrored, colorize, fps)

    def read(self):
        if self._current_frame < self._n_frames:
            depth_frame = cv2.imread(self._depth_paths[self._current_frame], cv2.IMREAD_ANYDEPTH)
            color_frame = cv2.imread(self._color_paths[self._current_frame])
            thermal_frame = cv2.imread(self._thermal_paths[self._current_frame], cv2.IMREAD_ANYDEPTH)
            self._current_frame += 1
            return depth_frame, color_frame, thermal_frame
        else:
            return None


