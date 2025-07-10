from ctcat import CTCAT_Sensor, CTCAT_DataFormat

CFG = {
    'device': 'linux-arm64',
    'sensor_id': 3
}

sensor = CTCAT_Sensor(
    sensor_id=CFG['sensor_id'],
    data_format=CTCAT_DataFormat.Resized,
    colorize=False,
    fps=None,
    device=CFG['device'],
)

