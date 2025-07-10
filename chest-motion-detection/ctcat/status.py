import time
import threading
from enum import Enum

GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except:
    print('Warning: Status Object not available (could not import RPi.GPIO)')

StatusColor = Enum('StatusColor', 'BLACK RED GREEN BLUE YELLOW MAGENTA CYAN WHITE')
StatusMode = Enum('StatusMode', 'OFF ON FLASHING')

class DummyStatus:
    def set(self, *args, **kwargs):
        pass

    def reset(self):
        pass

class StatusIndication:
    def __init__(self):
        self.pins = (32, 38, 40)  # red, green, blue pins
        self.color_cfg = {
            StatusColor.BLACK: (GPIO.LOW, GPIO.LOW, GPIO.LOW),
            StatusColor.RED: (GPIO.HIGH, GPIO.LOW, GPIO.LOW),
            StatusColor.GREEN: (GPIO.LOW, GPIO.HIGH, GPIO.LOW),
            StatusColor.BLUE: (GPIO.LOW, GPIO.LOW, GPIO.HIGH),
            StatusColor.YELLOW: (GPIO.HIGH, GPIO.HIGH, GPIO.LOW),
            StatusColor.MAGENTA: (GPIO.HIGH, GPIO.LOW, GPIO.HIGH),
            StatusColor.CYAN: (GPIO.LOW, GPIO.HIGH, GPIO.HIGH),
            StatusColor.WHITE: (GPIO.HIGH, GPIO.HIGH, GPIO.HIGH)
        }

        GPIO.setmode(GPIO.BOARD)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        # self.blink_interval = 0.1
        self.__color = StatusColor.BLACK
        self.__mode = StatusMode.OFF
        self.__time_on = 0.05
        self.__time_off = 1.

        self.__stop_event = threading.Event() # create the semaphore used to make thread exit
        self.__thread = threading.Thread(name='ledblink', target=self.__blink_pin)
        self.__thread.start()

    def __del__(self):
        self.reset()

    def set(self, mode=None, color=None, time_on=None, time_off=None):
        if mode is not None:
            self.__mode = mode

        if color is not None:
            self.__color = color

        if time_on is not None and time_on > 0:
            self.__time_on = time_on

        if time_off is not None and time_off >= 0:
            self.__time_off = time_off

    def reset(self):
        self.__stop_event.set() # set the semaphore so the thread will exit after sleep has completed
        self.__thread.join() # wait for the thread to exit
        GPIO.cleanup()

    def __set_on(self):
        for pin, state in zip(self.pins, self.color_cfg[self.__color]):
            GPIO.output(pin, state)

    def __set_off(self):
        for pin, state in zip(self.pins, self.color_cfg[StatusColor.BLACK]):
            GPIO.output(pin, state)

    def __blink_pin(self):
        while not self.__stop_event.is_set():
            # the first period is when the LED will be on if blinking
            if self.__mode == StatusMode.ON or self.__mode == StatusMode.FLASHING:
                self.__set_on()
            else:
                self.__set_off()

            time.sleep(self.__time_on)
            # only if blinking, turn led off and do a second sleep for the off time
            if self.__mode == StatusMode.FLASHING:
                self.__set_off()
                if not self.__stop_event.is_set(): # check stop semaphore again before off-time sleep
                    time.sleep(self.__time_off)


if GPIO_AVAILABLE:
    SaveStatusIndication = StatusIndication
else:
    SaveStatusIndication = DummyStatus

if __name__ == '__main__':
    status = StatusIndication()

    for color in StatusColor:
        status.set(color=color, mode=StatusMode.ON)
        print(f'color: {color}, mode: ON')
        time.sleep(5)
        status.set(color=color, mode=StatusMode.FLASHING)
        print(f'color: {color}, mode: FLASHING')
        time.sleep(5)

    status.reset()
