import os
import time
import traceback

import tqdm
import cv2
import flask
import numpy as np
from matplotlib import cm

from ctcat import CTCAT_Sensor, CTCAT_DataFormat
from status import SaveStatusIndication, StatusColor, StatusMode


def apply_colormap(array, cmap):
    lb, ub = array.min(), array.max()
    array = (array - lb) / (ub - lb + 1e-10)
    rgb_array = cm.get_cmap(cmap)(array)[..., :3]
    return rgb_array

def get_sequence_name():
    sequence_start = time.time()
    sequence_name = time.strftime('%d-%m-%Y_%H_%M_%S', time.localtime(sequence_start))
    return sequence_name, sequence_start

if __name__ == '__main__':
    cfg = {
        'recording_dir': '/media/cvl/niklas-ssd/recording', # <---- change this
        'device': 'linux-arm64',
        'sensor_id': 3
    }
    show = False
    register = False

    # initialize status led
    status = SaveStatusIndication()
    status.set(color=StatusColor.BLUE, mode=StatusMode.ON, time_off=0.)

    # initialize sensor
    data_format = CTCAT_DataFormat.Registered if register else CTCAT_DataFormat.Resized
    sensor = CTCAT_Sensor(sensor_id=cfg['sensor_id'], data_format=data_format, colorize=False, fps=None, device=cfg['device'])

    # setup logging
    if not show:
        sequence_name, sequence_start = get_sequence_name()
        sequence_dir = os.path.join(cfg['recording_dir'], sequence_name)
        if not os.path.exists(sequence_dir):
            os.makedirs(sequence_dir, exist_ok=True)
        print('recording directory created at %s' % sequence_dir)

        status.set(color=StatusColor.RED)


    try:
        for frame_number, data in tqdm.tqdm(enumerate(sensor)):
            if frame_number % 2 == 0:
                status.set(mode=StatusMode.ON)
            else:
                status.set(mode=StatusMode.OFF)

            # read next frame
            depth_frame, color_frame, thermal_frame = data

            # write frames to disk
            if not show:
                frame_time = int((time.time() - sequence_start) * 1000)
                frame_str = str(frame_number).zfill(6) + '_' + str(frame_time).zfill(10)
                cv2.imwrite(os.path.join(sequence_dir, 'depth_%s.png' % frame_str), depth_frame)
                cv2.imwrite(os.path.join(sequence_dir, 'rgb_%s.png' % frame_str), color_frame)
                cv2.imwrite(os.path.join(sequence_dir, 'thermal_%s.png' % frame_str), thermal_frame)

            if show:
                cv2.imshow('Recording', np.concatenate([
                    color_frame,
                    (apply_colormap(depth_frame, cmap='winter') * 255).astype(np.uint8)[..., ::-1],
                    (apply_colormap(thermal_frame, cmap='inferno') * 255).astype(np.uint8)[..., ::-1]
                ], axis=1))
                cv2.waitKey(1)
    except:
        print(traceback.format_exc())
        status.reset()
