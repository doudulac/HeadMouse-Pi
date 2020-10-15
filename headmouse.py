#!/usr/bin/env python3

# HeadMouse-Pi  --  Raspberry Pi based head tracking mouse
# Copyright (C) 2020  Kevin Rowland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import logging
import math
import multiprocessing as mp
import os
import queue
import shlex
import signal
import struct
import subprocess
import sys
import traceback
from os.path import getmtime
from threading import Thread

import cv2
import dlib
import numpy as np
import yappi
from filterpy.common.discretization import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from imutils import face_utils
from imutils.video import FPS
from scipy.linalg import block_diag


# MyVideoStream, MyPiVideoStream, and MyVideoCapture are derived from the 'imutils.video'
# package authored by Adrian Rosebrock and modified by Kevin Rowland. The following license applies:
#
# The MIT License (MIT)
#
# Copyright (c) 2015-2016 Adrian Rosebrock, http://www.pyimagesearch.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


class MyVideoStream:
    def __init__(self, src=0, usePiCamera=False, resolution=(320, 240), framerate=None,
                 frame_q=None, **kwargs):
        # check to see if the picamera module should be used
        if usePiCamera:
            # only import the picamera packages unless we are
            # explicitly told to do so -- this helps remove the
            # requirement of `picamera[array]` from desktops or
            # laptops that still want to use the `imutils` package

            # initialize the picamera stream and allow the camera
            # sensor to warmup
            self.stream = MyPiVideoStream(resolution=resolution, framerate=framerate,
                                          frame_q=frame_q, **kwargs)

        # otherwise, we are using OpenCV so initialize the webcam
        # stream
        else:
            self.stream = MyVideoCapture(src=src, resolution=resolution, framerate=framerate,
                                         frame_q=frame_q)

    def start(self):
        # start the threaded video stream
        return self.stream.start()

    def update(self):
        # grab the next frame from the stream
        self.stream.update()

    def read(self):
        # return the current frame
        return self.stream.read()

    def stop(self):
        # stop the thread and release any resources
        self.stream.stop()

    def join(self):
        self.stream.join()

    def fps(self):
        self.stream.fps()


class MyPiVideoStream:
    def __init__(self, resolution=(320, 240), framerate=30, frame_q=None, **kwargs):
        # initialize the camera
        import picamera.array
        resolution = picamera.PiResolution(resolution[0], resolution[1]).pad()
        self.camera = picamera.PiCamera(resolution=resolution, framerate=framerate)

        self.frame_q = frame_q

        # set optional camera parameters (refer to PiCamera docs)
        for (arg, value) in kwargs.items():
            setattr(self.camera, arg, value)

        # initialize the stream
        self.rawCapture = picamera.array.PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                                                     format="bgr", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.stopped = False

        self.thread = None
        self.frame = None
        self.framenum = None
        self.framew = None
        self.frameh = None
        self.skipped = 0

    def start(self):
        # start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        global running

        # keep looping infinitely until the thread is stopped
        framenum = 0
        try:
            for f in self.stream:
                # grab the frame from the stream and clear the stream in
                # preparation for the next frame
                frame = f.array
                if self.framew is None and frame is not None:
                    (self.frameh, self.framew) = frame.shape[:2]
                framenum += 1
                if self.frame_q is not None:
                    try:
                        self.frame_q.put_nowait((framenum, frame))
                    except queue.Full:
                        framenum -= 1
                        self.skipped += 1
                else:
                    self.frame = frame
                    self.framenum = framenum
                self.rawCapture.truncate(0)

                # if the thread indicator variable is set, stop the thread
                # and resource camera resources
                if self.stopped:
                    break
        except Exception as e:
            log.info("{}\n{}".format(e, traceback.format_exc()))
            running = False
            self.stopped = True

        self.stream.close()
        self.rawCapture.close()
        self.camera.close()
        if _args_.verbose > 0:
            log.info("Frames captured: {}".format(framenum))
            try:
                log.info("Frames orphaned: {}".format(self.frame_q.qsize()))
            except (NotImplementedError, AttributeError):
                pass
            log.info("Frames skipped: {}".format(self.skipped))

    def read(self):
        # return the frame most recently read
        if self.frame_q is not None:
            try:
                (self.framenum, self.frame) = self.frame_q.get(timeout=1)
            except queue.Empty:
                pass
        else:
            # time.sleep(.01)
            pass
        return self.framenum, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def join(self):
        self.thread.join()

    def fps(self):
        return self.camera.framerate


class MyVideoCapture(object):
    def __init__(self, src=0, resolution=(None, None), framerate=None, frame_q=None,
                 name="VideoCapture"):
        self.frame_q = frame_q
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        if resolution[0] is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        if resolution[1] is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if framerate is not None:
            self.stream.set(cv2.CAP_PROP_FPS, framerate)

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        self.thread = None
        self.frame = None
        self.framenum = None
        self.framew = None
        self.frameh = None
        self.skipped = 0

    def start(self):
        # start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, name=self.name, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        framenum = 0
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                if _args_.verbose > 0:
                    log.info("Frames captured: {}".format(framenum))
                    try:
                        log.info("Frames orphaned: {}".format(self.frame_q.qsize()))
                    except (NotImplementedError, AttributeError):
                        pass
                    log.info("Frames skipped: {}".format(self.skipped))

                return

            # otherwise, read the next frame from the stream
            (_, frame) = self.stream.read()
            if self.framew is None and frame is not None:
                (self.frameh, self.framew) = frame.shape[:2]
            framenum += 1
            if self.frame_q is not None:
                try:
                    self.frame_q.put_nowait((framenum, frame))
                except queue.Full:
                    framenum -= 1
                    self.skipped += 1
            else:
                self.framenum = framenum
                self.frame = frame

    def read(self):
        # return the frame most recently read
        if self.frame_q is not None:
            try:
                (self.framenum, self.frame) = self.frame_q.get(timeout=1)
            except queue.Empty:
                pass
        else:
            # time.sleep(.01)
            pass
        return self.framenum, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def join(self):
        self.thread.join()

    def fps(self):
        return self.stream.get(cv2.CAP_PROP_FPS)


class Face(object):
    def __init__(self, fps):
        self._angs_x = None
        self._angs_y = None
        self._cur_angle = [0, 0]
        self._ave_angle = None
        self._center = None
        self._shapes = None

        self.mouth = MouthOpen(face=self)
        self.eyes = Eyes(face=self, ear_threshold=_args_.ear, fps=fps)
        self.brows = Eyebrows(face=self, threshold=_args_.ebd, sticky=_args_.stickyclick)
        self.nose = Nose(face=self, use_kalman=_args_.filter, fps=fps)

    def update(self, shapes):
        self._shapes = shapes
        self._update_posture()
        self.mouth.update()
        self.eyes.update()
        self.brows.update()
        self.nose.update()

    def _update_posture(self):
        shapes = self.shapes
        if shapes is None:
            return

        # nose 27,30
        # nostr 31,35

        sep = feature_center(shapes[31:36])
        cen = feature_center(shapes)
        angle_x = cen[0] - sep[0]
        angle_y = cen[1] - sep[1]

        _n = 6
        _s = -2
        if self._angs_y is None:
            self._angs_x = [angle_x, ] * _n
            self._angs_y = [angle_y, ] * _n
        else:
            self._angs_x.append(self._angs_x.pop(0))
            self._angs_x[-1] = angle_x
            self._angs_y.append(self._angs_y.pop(0))
            self._angs_y[-1] = angle_y

        self._cur_angle = [sum(self._angs_x[_s:]) / len(self._angs_x[_s:]),
                           sum(self._angs_y[_s:]) / len(self._angs_y[_s:])]
        self._ave_angle = [sum(self._angs_x[:_s]) / len(self._angs_x[:_s]),
                           sum(self._angs_y[:_s]) / len(self._angs_y[:_s])]
        self._center = cen

        if _args_.debug_face:
            log.info("face {:6.02f} {:6.02f} {:6.02f} {:6.02f}".format(self._ave_angle[0],
                                                                       self._cur_angle[0],
                                                                       self._ave_angle[1],
                                                                       self._cur_angle[1]))

    def facing_camera(self):
        if abs(self.x_angle) > 5.0 or self.y_angle < -5.0 or self.y_angle > 0:
            return False
        else:
            return True

    @property
    def shapes(self):
        return self._shapes

    @property
    def center(self):
        return self._center

    @property
    def angle(self):
        return self._cur_angle

    @property
    def ave_angle(self):
        return self._ave_angle

    @property
    def x_angle(self):
        return self._cur_angle[0]

    @property
    def x_ave_angle(self):
        return self._ave_angle[0]

    @property
    def y_angle(self):
        return self._cur_angle[1]

    @property
    def y_ave_angle(self):
        return self._ave_angle[1]


class MouthOpen(object):
    def __init__(self, face):
        self.face = face
        self._vdists = None
        self._cur_vdist = None
        self._hdists = None
        self._cur_hdist = None
        self._open = False

    def update(self):
        shapes = self.face.shapes
        if shapes is None:
            return

        # moutho 48,59
        # mouthi 60,67

        lc = shapes[60]
        rc = shapes[64]
        hdist = point_distance(lc, rc)
        up = shapes[62]
        lo = shapes[66]
        vdist = point_distance(up, lo)

        _n = 6
        _s = -2
        if self._vdists is None:
            self._vdists = [vdist, ] * _n
            self._hdists = [hdist, ] * _n
        else:
            self._vdists.append(self._vdists.pop(0))
            self._vdists[-1] = vdist
            self._hdists.append(self._hdists.pop(0))
            self._hdists[-1] = hdist

        self._cur_vdist = sum(self._vdists[_s:]) / len(self._vdists[_s:])
        vpast = sum(self._vdists[:_s]) / len(self._vdists[:_s])
        self._cur_hdist = sum(self._hdists[_s:]) / len(self._hdists[_s:])
        hpast = sum(self._hdists[:_s]) / len(self._hdists[:_s])
        _r = self._cur_vdist / self._cur_hdist

        if not self._open:
            self._open = _r >= .50
        else:
            self._open = _r >= .40

        if _args_.debug_mouth:
            log.info("mouth {:.02f} {:.02f} {:.02f} {:.02f} {:.02f}".format(
                vpast, self._cur_vdist, hpast, self._cur_hdist, _r))

    @property
    def open(self):
        return self._open

    def button_up(self):
        return self._open is False

    def button_down(self):
        return self._open is True


class Eyebrows(object):
    def __init__(self, face, threshold, sticky=False):
        self.face = face
        self.threshold = threshold

        self._ebds = None
        self._cur_height = None
        self._ave_height = None
        self._raised = False
        self._raised_count = 0
        self.sticky = sticky
        self._sticky_raised = False

    def update(self):
        shapes = self.face.shapes
        if shapes is None:
            return

        # jaw 0,16
        # rbrow 17,21
        # lbrow 22,26
        # nose 27,30
        # nostr 31,35
        # reye 36,41
        # leye 42,47
        # moutho 48,59
        # mouthi 60,67

        leye = feature_center(shapes[42:48])
        reye = feature_center(shapes[36:42])
        pdist = point_distance(leye, reye)
        ebc = feature_center([shapes[19], shapes[24]])
        eyec = feature_center(shapes[36:48])
        ebd = point_distance(ebc, eyec)
        _r = ebd / pdist

        _n = 6
        _s = -2
        if self._ebds is None:
            self._ebds = [ebd, ] * _n
        else:
            self._ebds.append(self._ebds.pop(0))
            self._ebds[-1] = ebd

        self._cur_height = sum(self._ebds[_s:]) / len(self._ebds[_s:])
        self._ave_height = sum(self._ebds[:_s]) / len(self._ebds[:_s])

        d_angle = self.face.y_angle - self.face.y_ave_angle
        if self.sticky:
            raised = self._cur_height - self._ave_height > self.threshold and d_angle < 2.0
            if not self._raised:
                self._raised = raised
            else:
                d_angle = self.face.y_ave_angle - self.face.y_angle
                lowered = self._ave_height - self._cur_height > self.threshold * .60 and \
                    d_angle < 2.0
                if not self._sticky_raised or self._raised_count > 0:
                    self._raised = not lowered
                elif raised:
                    self._sticky_raised = False

            if self._raised and not self._sticky_raised:
                self._raised_count += 1
                if self._raised_count > int(round(_fps_.fps() * .500)):
                    self._sticky_raised = True
                    self._raised_count = 0
            else:
                self._raised_count = 0

        else:
            self._raised = self._cur_height - self._ave_height > self.threshold and d_angle < 2.0

        if _args_.debug_brows:
            line = "brows {:.02f} {:.02f} {:.02f} {:.02f} {:.02f}"
            log.info(line.format(self._ave_height, self._cur_height, _r, pdist, ebd))

    def reset(self):
        self._raised = False
        self._raised_count = 0
        self._sticky_raised = False

    @property
    def cur_height(self):
        return self._cur_height

    @property
    def ave_height(self):
        return self._ave_height

    @property
    def raised(self):
        return self._raised

    def button_up(self):
        return self._raised is False

    def button_down(self):
        return self._raised is True


class Eyes(object):
    def __init__(self, face, ear_threshold=None, fps=None):
        self.face = face
        self._open = False
        if ear_threshold is None:
            ear_threshold = .15
        self._ear_threshold = ear_threshold
        if fps is None:
            fps = 20
        dt = 1.0 / fps
        self._kf = kalmanfilter_dim2_init(dt=dt, Q=1 ** 2, R=.05)
        self._pupilary_dist = None

    def update(self):
        shapes = self.face.shapes
        if shapes is None:
            return

        # reye 36,41
        # leye 42,47

        rec = feature_center(shapes[36:42])
        lec = feature_center(shapes[42:48])
        self._pupilary_dist = point_distance(rec, lec)

        red = (point_distance(shapes[37], shapes[41]) + point_distance(shapes[38], shapes[40])) / \
              (2.0 * point_distance(shapes[36], shapes[39]))
        led = (point_distance(shapes[43], shapes[47]) + point_distance(shapes[44], shapes[46])) / \
              (2.0 * point_distance(shapes[42], shapes[45]))
        ear = (red + led) / 2

        self._kf.predict()
        self._kf.update(ear)
        if self._kf.x[0][0] < self._ear_threshold:
            self._open = False
        else:
            self._open = True

        if _args_.debug_eyes:
            log.info("eyes {:5.02f} {:5.02f} {:6.03f} {}".format(
                ear, self._kf.x[0][0], self._kf.x[1][0], "O" if self._open else "."
            ))

    @property
    def open(self):
        return self._open

    @property
    def pupilary_dist(self):
        return self._pupilary_dist

    def button_up(self):
        return self.open

    def button_down(self):
        return not self.button_up()


class Nose(object):
    def __init__(self, face, use_kalman=False, fps=None):
        self.face = face
        self._positions = None
        self.nose_raw = [0, 0]
        self._dx = None
        self._dy = None
        self._vels = None
        self._ax = None
        self._ay = None
        if use_kalman:
            if fps is None:
                fps = 20
            dt = 1.0 / fps
            self._kf = kalmanfilter_dim4_init(dt=dt, Q=10 ** 2, R=.05)
        else:
            self._kf = None

    def update(self):
        # jaw 0,16
        # rbrow 17,21
        # lbrow 22,26
        # nose 27,30
        # nostr 31,35
        # reye 36,41
        # leye 42,47
        # moutho 48,59
        # mouthi 60,67
        shapes = self.face.shapes
        if shapes is not None:
            self.nose_raw = shapes[30]
            kfu = self.nose_raw
        else:
            # Let self.nose_raw from last time persist, making dx = 0 if no filter
            kfu = None

        if self._kf is None:
            nose = self.nose_raw
            vel = [0, 0]
        else:
            self._kf.predict()
            self._kf.update(kfu)
            nose = [int(round(self._kf.x[0][0])), int(round(self._kf.x[2][0]))]
            vel = [self._kf.x[1][0], self._kf.x[3][0]]

        if self._positions is None:
            self._positions = [nose, nose]
            self._vels = [vel, vel]
        else:
            self._positions.append(self._positions.pop(0))
            self._positions[-1] = nose
            self._vels.append(self._vels.pop(0))
            self._vels[-1] = vel

        self._dx = self.position[0] - self.prev_position[0]
        self._dy = self.position[1] - self.prev_position[1]
        self._ax = self.vel[0] - self.prev_vel[0]
        self._ay = self.vel[1] - self.prev_vel[1]

        if _args_.debug_nose:
            line = "nose ({:7.03f}, {:7.03f}) ({:8.03f}, {:8.03f}) ({:8.03f}, {:8.03f}) "
            line += "({:7.03f}, {:6.02f}) ({:7.03f}, {:6.02f})"
            log.info(line.format(self.nose_raw[0], self.nose_raw[1], self.position[0], self.vel[0],
                                 self.position[1], self.vel[1], self.dx, self.ax, self.dy, self.ay))

    @property
    def ax(self):
        return self._ax

    @property
    def ay(self):
        return self._ay

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def prev_position(self):
        return self._positions[0]

    @property
    def position(self):
        return self._positions[1]

    @property
    def prev_vel(self):
        return self._vels[0]

    @property
    def vel(self):
        return self._vels[1]

    @property
    def using_kfilter(self):
        return self._kf is not None


class MousePointer(object):
    def __init__(self, face, mindeltathresh=None):
        self.face = face
        self._fd = None
        self.open_hidg()
        self._dx = None
        self._dy = None
        self._click = None
        self.btn = {1: {'s': 0, 'f': self.face.brows},
                    2: {'s': 0, 'f': self.face.mouth},
                    3: {'s': 0, 'f': None}, }
        self.pausebtn = {'s': 0, 'f': self.face.eyes}
        self._paused = False
        self.cpos = None
        self.angle = None
        self.track_cpos = self._fd is None
        self.maxheight = None
        self.maxwidth = None
        self.wrap = False
        self.xgain = _args_.xgain if _args_.xgain is not None else 1.0
        self.ygain = _args_.ygain if _args_.ygain is not None else 1.0
        self.mindeltathresh = mindeltathresh if mindeltathresh is not None else 1
        self._smoothness = None
        self._motionweight = None
        self.set_smoothness(_args_.smoothness)
        self._prevdx = 0
        self._prevdy = 0
        self.i_accel = None
        # self.accel = [0.0, 1.0, 1.2, 1.4, 1.6,
        #               1.8, 2.0, 2.2, 2.4, 2.6,
        #               2.3, 2.4, 2.5, 3.0, 4.0,
        #               4.1, 4.2, 4.3, 4.4, 4.5,
        #               4.6, 4.7, 4.8, 4.9, 5.0,
        #               5.1, 5.2, 5.3, 5.4, 5.5, ]
        self.accel = [1.0]*10 + [2.0]*20

    def open_hidg(self):
        if self._fd is not None:
            return self._fd
        try:
            self._fd = open('/dev/hidg0', 'r+b', buffering=0)
        except FileNotFoundError:
            self._fd = None

    def close_hidg(self):
        if self._fd is not None:
            self._fd.close()
            self._fd = None

    def process_movement(self):
        nose = self.face.nose
        dx = nose.dx
        dy = nose.dy
        dx *= self.xgain
        dy *= self.ygain
        if not nose.using_kfilter:
            dx = dx * (1.0 - self._motionweight) + self._prevdx * self._motionweight
            dy = dy * (1.0 - self._motionweight) + self._prevdy * self._motionweight
            self._prevdx = dx
            self._prevdy = dy

        dist = math.sqrt(dx * dx + dy * dy)
        self.i_accel = int(dist + 0.5)
        if self.i_accel >= len(self.accel):
            self.i_accel = len(self.accel) - 1

        if not nose.using_kfilter:
            if -self.mindeltathresh < dx < self.mindeltathresh:
                dx = 0
            if -self.mindeltathresh < dy < self.mindeltathresh:
                dy = 0
        dx *= self.accel[self.i_accel]
        dy *= self.accel[self.i_accel]
        dx = -int(round(dx))
        dy = int(round(dy))

        self._dx = dx
        self._dy = dy

        if self.track_cpos:
            try:
                self.cpos[0] += dx
                self.cpos[1] += dy
            except TypeError:
                self.cpos = [int(self.maxwidth/2), int(self.maxheight/2)]

            if self.cpos[0] > self.maxwidth:
                if self.wrap:
                    self.cpos[0] -= self.maxwidth
                else:
                    self.cpos[0] = self.maxwidth
            if self.cpos[1] > self.maxheight:
                if self.wrap:
                    self.cpos[1] -= self.maxheight
                else:
                    self.cpos[1] = self.maxheight
            if self.cpos[0] < 0:
                if self.wrap:
                    self.cpos[0] += self.maxwidth
                else:
                    self.cpos[0] = 0
            if self.cpos[1] < 0:
                if self.wrap:
                    self.cpos[1] += self.maxheight
                else:
                    self.cpos[1] = 0

        if _args_.debug_mouse:
            log.info("mouse {} {:2} {}".format(self.cpos, self.i_accel, self.accel[self.i_accel]))

        return dx, dy

    def process_clicks(self):
        click = 0

        for i, btn in self.btn.items():
            if btn['f'] is None:
                continue
            if btn['s'] > 0:
                btn['s'] -= 1
                if btn['s'] == 0:
                    if btn['f'].button_down():
                        if i == 1 and _args_.stickyclick:
                            btn['s'] = -1
                        if i == 2:
                            btn['s'] = 1
            elif btn['s'] < 0:
                if btn['f'].button_up():
                    btn['s'] = 0

            if btn['s'] < 0:
                click |= 1 << (i - 1)
            elif btn['s'] == 0 and btn['f'].button_down():
                if i != 1:
                    btn['s'] = int(round(_fps_.fps() * .4))
                click |= 1 << (i - 1)

        self._click = click

        return click

    def process_pause(self):
        btn = self.pausebtn

        if btn['f'] is None:
            return

        if not self.face.facing_camera():
            btn['s'] = 0
            return

        if btn['s'] > 0:
            btn['s'] -= 1
            if btn['f'].button_up():
                # reset
                btn['s'] = 0
            elif btn['s'] == 0:
                # wait for button up to activate
                btn['s'] = -1
        elif btn['s'] < 0:
            if btn['f'].button_up():
                btn['s'] = 0
                self._paused = not self._paused
                self.face.brows.reset()
                if _args_.verbose > 0:
                    log.info('{}pause'.format('' if self.paused else 'un'))

        if btn['s'] == 0 and btn['f'].button_down():
            btn['s'] = int(round(_fps_.fps() * 1.5))

    def update(self):
        self.process_pause()
        if not self.paused:
            dx, dy = self.process_movement()
            click = self.process_clicks()
            self.send_mouse_relative(click, dx, dy)
        elif self.click != 0:
            self._click = 0
            self.send_mouse_relative(0, 0, 0)
        elif self.face.shapes is not None:
            x, y = self.process_circle_pattern()
            self.send_mouse_absolute(x, y, 0)

    def send_mouse_relative(self, click, dx, dy):
        if _args_.verbose > 1 and click:
            log.info('click {}'.format(click))

        if self._fd is not None:
            report = struct.pack('<2b2h', 2, click, dx, dy)
            self._fd.write(report)

    def send_mouse_absolute(self, _x, _y, wheel):
        if self._fd is not None:
            report = struct.pack('<b2hb', 3, _x, _y, wheel)
            self._fd.write(report)

    def set_smoothness(self, value):
        if value is None or value < 0:
            value = 0
        elif value > 8:
            value = 8

        self._smoothness = value
        self._motionweight = math.log10(float(self._smoothness) + 1)

    def process_circle_pattern(self):
        radius = self.maxwidth / 32
        if self.face.facing_camera():
            omega = 1.75 * math.radians(360) / _fps_.fps()
        else:
            omega = .75 * math.radians(360) / _fps_.fps()

        try:
            x = self.cpos[0]
            y = self.cpos[1]
            self.angle = self.angle + omega
            x = x + radius * omega * math.cos(self.angle + math.pi / 2)
            y = y - radius * omega * math.sin(self.angle + math.pi / 2)
        except TypeError:
            center = [self.maxwidth / 2, self.maxheight / 2]
            self.angle = math.radians(90)
            x = center[0] + radius * math.cos(self.angle)
            y = center[1] - radius * math.sin(self.angle)

        x, y = int(round(x)), int(round(y))
        self.cpos = [x, y]

        return x, y

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def click(self):
        return self._click

    @property
    def paused(self):
        return self._paused

    @property
    def smoothness(self):
        return self._smoothness

    @property
    def motionweight(self):
        return self._motionweight


class MyLogger(object):
    def __init__(self):
        logging.raiseExceptions = False
        self._root = logging.root
        for h in self._root.handlers:
            self._root.removeHandler(h)
        self._hndlr = logging.StreamHandler(stream=sys.stdout)
        self._fmt = logging.Formatter(fmt='{asctime}| {message}', datefmt='%Y%m%d %H:%M:%S',
                                      style='{')
        self._hndlr.setFormatter(self._fmt)
        self._root.addHandler(self._hndlr)
        self._root.setLevel(logging.WARNING)

        self._ends = []

    def setLevel(self, level):
        self._root.setLevel(level)

    def log(self, level, msg, *args, **kwargs):
        if len(self._ends):
            if self._hndlr.formatter is not None:
                self._hndlr.formatter = None
        elif self._hndlr.formatter is None:
            self._hndlr.formatter = self._fmt

        popend = kwargs.pop('popend', False)
        if popend:
            self._hndlr.terminator = self._ends.pop()

        end = kwargs.pop('end', None)
        if end is not None:
            self._ends.append(self._hndlr.terminator)
            self._hndlr.terminator = end

        try:
            self._root.log(level, msg, *args, **kwargs)
        except BrokenPipeError:
            pass

    def debug(self, msg, end=None, popend=False, *args, **kwargs):
        self.log(logging.DEBUG, msg, end, popend, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(logging.INFO, msg, *args, **kwargs)


def annotate_frame(frame, shapes, face, mouse):
    nose = face.nose
    brows = face.brows
    if mouse.cpos is not None:
        cv2.circle(frame, (int(mouse.cpos[0]), int(mouse.cpos[1])), 4, (0, 0, 255), -1)
        cv2.putText(frame, "ptr : " + str((int(mouse.cpos[0]), int(mouse.cpos[1]))), (90, 270),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.putText(frame, "brows: {:.2f} {}".format(brows.cur_height, brows.raised), (90, 165),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "nose: " + str(nose.position), (90, 200),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "dxdy: " + str((mouse.dx, mouse.dy)), (90, 235),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    draw_landmarks(frame, shapes, face.center)
    # frame = face_utils.visualize_facial_landmarks(frame, shapes, [(0,255,0),]*8)
    return frame


def draw_landmarks(frame, shapes, center):
    if shapes is None:
        return

    cv2.circle(frame, center, 4, (128, 0, 255), -1)
    for (i, j) in face_utils.FACIAL_LANDMARKS_68_IDXS.values():
        pts = shapes[i:j]
        for _l in range(1, len(pts)):
            p1 = tuple(pts[_l - 1])
            p2 = tuple(pts[_l])
            cv2.line(frame, p1, p2, (0, 255, 0))
            # if _l == 1:
            #     cv2.line(frame, center, p1, (0, 255, 255))
            # cv2.line(frame, center, p2, (0, 255, 255))


def face_detect_mp(frameq, shapesq, detector, predictor, args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    r = None
    dim = None
    firstframe = True
    for framenum, frame in iter(frameq.get, "STOP"):
        if frame is None:
            shapesq.put_nowait((framenum, None, None))
            continue

        if firstframe:
            firstframe = False
            (h, w) = frame.shape[:2]
            r = 320 / float(w)
            dim = (320, int(h * r))
            if args.verbose > 0 and mp.current_process().name[-2:] == "-1":
                msg = " Frame shape: {}".format(frame.shape)
                shapesq.put_nowait((-1, msg, mp.current_process().name))
                msg = "Frame scaled: {}".format(dim)
                shapesq.put_nowait((-1, msg, mp.current_process().name))

        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sgframe = cv2.resize(gframe, dim, interpolation=cv2.INTER_AREA)
        faces = detector(sgframe)
        if len(faces) == 0:
            shapes = None
        else:
            face_rect = dlib.rectangle(int(faces[0].left() / r),
                                       int(faces[0].top() / r),
                                       int(faces[0].right() / r),
                                       int(faces[0].bottom() / r))
            shapes = predictor(gframe, face_rect)
            shapes = face_utils.shape_to_np(shapes)
        if args.onraspi and not _args_.debug_video:
            frame = None
        shapesq.put_nowait((framenum, frame, shapes))

    shapesq.cancel_join_thread()

    return 0


def face_detect(demoq, detector, predictor):
    global running
    global restart
    global _args_
    global _fps_

    _fps_ = FPS()
    rotate = None
    picam = _args_.onraspi and not _args_.usbcam
    frameq = queue.Queue(5) if _args_.qmode else None
    resolution = (1280, 720)
    framerate = 30
    if _args_.onraspi and not _args_.usbcam:
        if _args_.camera_mode == 1:
            resolution = (320, 240)
        elif _args_.camera_mode == 2:
            resolution = (640, 480)
        elif _args_.camera_mode == 3:
            resolution = (800, 600)
            framerate = 15
        elif _args_.camera_mode == 4:
            resolution = (1024, 768)
            framerate = 10
        rotate = 180

    cam = MyVideoStream(usePiCamera=picam, resolution=resolution, framerate=framerate,
                        frame_q=frameq, rotation=rotate).start()

    if _args_.debug_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter("/tmp/testfile.avi", fourcc, framerate, resolution)
    else:
        writer = None

    face = Face(fps=framerate)
    mouse = MousePointer(face=face)
    if _args_.debug:
        mouse.close_hidg()

    wd = int(int(os.getenv('WATCHDOG_USEC', 0)) / 1000000)
    if wd > 0:
        import systemd.daemon as daemon
        log.info("watchdog: {}".format(wd))
        daemon.notify("READY=1\nWATCHDOG=1")

    no_face_frames = 0
    r = None
    dim = None
    framenum = 0
    firstframe = True
    mtime = getmtime(__file__)
    while running:
        pframenum = framenum
        (framenum, frame) = cam.read()
        if frame is None or framenum == pframenum:
            continue

        if firstframe:
            firstframe = False
            _fps_.start()
            (h, w) = frame.shape[:2]
            mouse.maxheight, mouse.maxwidth = (h, w)
            r = 320 / float(w)
            dim = (320, int(h * r))
            if _args_.verbose > 0:
                log.info(" Frame shape: {}".format(frame.shape))
                log.info("Frame scaled: {}".format(dim))

        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sgframe = cv2.resize(gframe, dim, interpolation=cv2.INTER_AREA)
        faces = detector(sgframe)
        if len(faces) == 0:
            if not mouse.paused:
                no_face_frames += 1
                if _args_.verbose > 0:
                    log.debug('{} no face'.format(no_face_frames))
            continue

        face_rect = dlib.rectangle(int(faces[0].left() / r),
                                   int(faces[0].top() / r),
                                   int(faces[0].right() / r),
                                   int(faces[0].bottom() / r))
        shapes = predictor(gframe, face_rect)
        shapes = face_utils.shape_to_np(shapes)

        _fps_.stop()
        try:
            if wd > 0 and framenum % int(wd / 4 * _fps_.fps()) == 0:
                daemon.notify("WATCHDOG=1")
                log.debug('{}: wd({})'.format(framenum, wd))

            if framenum > 5 and framenum % int(2 * _fps_.fps()) == 0:
                if getmtime(__file__) != mtime:
                    restart = True
                    running = False
                    break
        except ZeroDivisionError:
            _fps_.start()

        face.update(shapes)
        try:
            mouse.update()
        except BrokenPipeError as e:
            if _args_.verbose > 0:
                log.info(e)
            restart = True
            running = False
            break

        if not _args_.onraspi or writer is not None:
            annotate_frame(frame, shapes, face, mouse)

        if writer is not None:
            writer.write(frame)

        if not _args_.onraspi:
            demoq.put_nowait(frame)

        _fps_.update()

    if wd > 0:
        msg = "RELOADING=1" if restart else "STOPPING=1"
        daemon.notify(msg)

    if _args_.verbose > 0:
        log.info("Shutting down ...")

    if writer is not None:
        writer.release()

    mouse.send_mouse_relative(0, 0, 0)
    mouse.close_hidg()

    _fps_.stop()
    cfps = cam.fps()
    cam.stop()
    cam.join()

    if _args_.verbose > 0:
        log.info("Elapsed time: {:.1f}s".format(_fps_.elapsed()))
        log.info("         FPS: {:.3f}/{}".format(_fps_.fps(), cfps))
        log.info("     No face: {}".format(no_face_frames))


def feature_center(shapes):
    centx, centy = 0, 0
    if shapes is not None:
        for x, y in shapes:
            centx += x
            centy += y
        centx = int(centx / len(shapes))
        centy = int(centy / len(shapes))
    return centx, centy


def kalmanfilter_dim4_init(dt=1 / 20, Q=2.0, R=2.0):
    f = KalmanFilter(dim_x=4, dim_z=2)
    # State Transition matrix
    f.F = np.array([[1., dt, 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., dt],
                    [0., 0., 0., 1.]])
    # Process noise matrix
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    f.Q = block_diag(q, q)
    # Measurement function
    f.H = np.array([[1., 0., 0., 0.],
                    [0., 0., 1., 0.]])
    # Measurement noise matrix
    f.R = np.array([[R, 0.],
                    [0., R]])
    # Current state estimate
    f.x = np.array([[0., 0., 0., 0.]]).T
    # Current state covariance matrix
    f.P = np.eye(4) * 1000.
    return f


def kalmanfilter_dim2_init(dt=1 / 20, Q=2.0, R=2.0):
    f = KalmanFilter(dim_x=2, dim_z=1)
    # State Transition matrix
    f.F = np.array([[1., dt],
                    [0., 1.]])
    # Process noise matrix
    f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    # Measurement function
    f.H = np.array([[1., 0.]])
    # Measurement noise matrix
    f.R = np.array([[R]])
    # Current state estimate
    f.x = np.array([[0., 0.]]).T
    # Current state covariance matrix
    f.P = np.eye(2) * 1000.
    return f


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera-mode", default=2, type=int, choices=range(1, 5),
                        help="camera mode 1:320x240@30 2:640x480@30 3:800x600@15 4:1024x768@10" +
                             " (default: 2)")
    parser.add_argument("-e", "--ebd", default=4.0, type=float,
                        help="Eyebrow distance for click (default: 4.0)")
    parser.add_argument("-f", "--filter", action="store_true",
                        help="enable filter")
    parser.add_argument("-k", "--stickyclick", action="store_true",
                        help="enable eyebrow sticky click")
    parser.add_argument("-p", "--profile", action="store_true",
                        help="enable profiling")
    parser.add_argument("-q", "--qmode", action="store_true",
                        help="enable queue mode")
    parser.add_argument("-r", "--procs", default=5, type=int,
                        help="number of procs (default: 5)")
    parser.add_argument("-s", "--smoothness", default=3, type=int, choices=range(1, 9),
                        help="smoothness 1-8 (default: 3)")
    parser.add_argument("-u", "--usbcam", action="store_true",
                        help="Use usb camera instead of PiCamera")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Verbosity")
    parser.add_argument("-x", "--xgain", default=3.5, type=float,
                        help="X gain")
    parser.add_argument("-y", "--ygain", default=4.5, type=float,
                        help="Y gain")
    parser.add_argument("--ear", default=.15,
                        help="eye aspect ratio threshold (default: .15)")
    parser.add_argument("--onraspi", action="store_true",
                        help="force raspi mode")
    parser.add_argument("--debug", action="store_true",
                        help="disable mouse reports to host")
    parser.add_argument("--debug-video", action="store_true",
                        help="save video to testfile")
    parser.add_argument("--debug-face", action="store_true",
                        help="")
    parser.add_argument("--debug-brows", action="store_true",
                        help="")
    parser.add_argument("--debug-eyes", action="store_true",
                        help="")
    parser.add_argument("--debug-mouth", action="store_true",
                        help="")
    parser.add_argument("--debug-nose", action="store_true",
                        help="")
    parser.add_argument("--debug-mouse", action="store_true",
                        help="")

    args = parser.parse_args()
    return args


def point_distance(p1, p2):
    d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return d


def renice(nice, pids):
    if not _args_.onraspi:
        return

    if isinstance(pids, int):
        _pids = [pids, ]
    else:
        _pids = pids

    if _args_.verbose:
        log.info("Adjusting process priority.")
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
    else:
        stdout = subprocess.DEVNULL
        stderr = stdout

    pstr = " ".join(["-p {}".format(p) for p in _pids])
    cmd = shlex.split("/usr/bin/sudo /usr/bin/renice {} {}".format(nice, pstr))
    p = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    try:
        outs = p.communicate(timeout=5)[0]
    except TimeoutError:
        p.kill()
        outs = p.communicate()[0]

    if _args_.verbose:
        outs = outs.decode('utf8').rstrip('\n')
        log.info(outs)


def start_face_detect_procs(detector, predictor):
    global _fps_
    global restart
    global running

    if _args_.verbose > 0:
        log.info("{} pid {}".format(mp.current_process().name, mp.current_process().pid))
    frameq = mp.Queue(5)
    shapesqs = [mp.Queue()] * _args_.procs
    workers = []
    for i in range(_args_.procs):
        _p = mp.Process(target=face_detect_mp, args=(frameq, shapesqs[i], detector, predictor,
                                                     _args_))
        _p.start()
        workers.append(_p)
        if _args_.verbose > 0:
            log.info("{} pid {}".format(_p.name, _p.pid))

    _fps_ = FPS()
    rotate = None
    picam = _args_.onraspi and not _args_.usbcam
    resolution = (1280, 720)
    framerate = 30
    if _args_.onraspi and not _args_.usbcam:
        if _args_.camera_mode == 1:
            resolution = (320, 240)
        elif _args_.camera_mode == 2:
            resolution = (640, 480)
        elif _args_.camera_mode == 3:
            resolution = (800, 600)
            framerate = 15
        elif _args_.camera_mode == 4:
            resolution = (1024, 768)
            framerate = 10
        rotate = 180

    cam = MyVideoStream(usePiCamera=picam, resolution=resolution, framerate=framerate,
                        frame_q=frameq, rotation=rotate).start()

    if _args_.debug_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter("/tmp/testfile.avi", fourcc, framerate, resolution)
    else:
        writer = None

    face = Face(fps=framerate)
    mouse = MousePointer(face=face)
    if _args_.debug:
        mouse.close_hidg()

    wd = int(int(os.getenv('WATCHDOG_USEC', 0)) / 1000000)
    if wd > 0:
        import systemd.daemon as daemon
        log.info("watchdog: {}".format(wd))
        daemon.notify("READY=1\nWATCHDOG=1")

    no_face_frames = 0
    firstframe = True
    timeout = None
    mtime = getmtime(__file__)
    ooobuf = {}
    nextframe = 1
    qnum = -1
    try:
        while running:
            try:
                framenum, frame, shapes = ooobuf[nextframe]
            except KeyError:
                try:
                    qnum += 1
                    if qnum >= len(shapesqs):
                        qnum = 0
                    framenum, frame, shapes = shapesqs[qnum].get(timeout=timeout)
                except queue.Empty:
                    if _args_.verbose > 0 and framerate - _fps_.fps() > 2:
                        log.info("queue delay, fps[{:.02f}]...low voltage?".format(_fps_.fps()))
                    continue
            if framenum != nextframe:
                if framenum < 0 and _args_.verbose >= abs(framenum):
                    log.info(frame)
                else:
                    ooobuf[framenum] = (framenum, frame, shapes)
                continue
            nextframe += 1

            if shapes is None and not mouse.paused:
                no_face_frames += 1
                if _args_.verbose > 0:
                    log.debug('{} no face'.format(no_face_frames))

            if firstframe:
                firstframe = False
                timeout = 3 / framerate
                if _args_.onraspi:
                    mouse.maxwidth, mouse.maxheight = 32767, 32767
                else:
                    mouse.maxwidth, mouse.maxheight = cam.framew, cam.frameh
                _fps_.start()

            _fps_.stop()
            try:
                if wd > 0 and framenum % int(wd / 4 * _fps_.fps()) == 0:
                    daemon.notify("WATCHDOG=1")
                    log.debug('{}: wd({})'.format(framenum, wd))

                if framenum > 5 and framenum % int(2 * _fps_.fps()) == 0:
                    if getmtime(__file__) != mtime:
                        restart = True
                        break
            except ZeroDivisionError:
                _fps_.start()

            face.update(shapes)
            mouse.update()

            if not _args_.onraspi or writer is not None:
                annotate_frame(frame, shapes, face, mouse)

            if writer is not None:
                writer.write(frame)

            if not _args_.onraspi:
                cv2.imshow("Demo", frame)

                if cv2.waitKey(1) == 27:
                    break

            _fps_.update()
    except Exception as e:
        log.info("{}\n{}".format(e, traceback.format_exc()))

    if wd > 0:
        msg = "RELOADING=1" if restart else "STOPPING=1"
        daemon.notify(msg)

    if _args_.verbose > 0:
        log.info("Shutting down ...")

    if writer is not None:
        writer.release()

    mouse.send_mouse_relative(0, 0, 0)
    mouse.close_hidg()

    _fps_.stop()
    cfps = cam.fps()
    cam.stop()
    cam.join()

    for i in range(_args_.procs):
        frameq.put('STOP')
    for p in workers:
        if _args_.verbose > 0:
            log.info("Joining {}...".format(p.name), end="")
        p.join()
        if _args_.verbose > 0:
            log.info("stopped.", popend=True)

    if _args_.verbose > 0:
        log.info("Elapsed time: {:.1f}s".format(_fps_.elapsed()))
        log.info("         FPS: {:.3f}/{}".format(_fps_.fps(), cfps))
        log.info("     No face: {}".format(no_face_frames))


def start_face_detect_thread(detector, predictor):
    global running

    fps = FPS()
    demoq = queue.Queue()
    t = Thread(target=face_detect, args=(demoq, detector, predictor))
    t.start()
    firstframe = True
    while running:
        try:
            frame = demoq.get(timeout=1)
        except queue.Empty:
            continue
        if firstframe:
            firstframe = False
            fps.start()

        cv2.imshow("Demo", frame)
        fps.update()

        if cv2.waitKey(1) == 27:
            running = False

    fps.stop()
    t.join()

    if _args_.verbose > 0 and not _args_.onraspi:
        log.info("Demo Queue")
        log.info("Elapsed time: {:.1f}s".format(fps.elapsed()))
        log.info("         FPS: {:.3f}".format(fps.fps()))


def sig_handler(signum, _frame):
    global running

    if _args_.verbose > 0:
        log.info("Caught signal '{}'".format(signum))

    if signum != signal.SIGPIPE:
        running = False


def main():
    global running
    global restart
    global _args_

    _args_ = parse_arguments()

    if _args_.verbose > 2:
        _args_.verbose = 2

    level = -_args_.verbose * logging.DEBUG + logging.WARNING
    log.setLevel(level)

    if _args_.verbose > 0:
        log.info("Starting {}".format(' '.join(sys.argv)))

    if _args_.debug and _args_.verbose < 2:
        _args_.verbose = 2

    if not _args_.onraspi:
        try:
            with open('/dev/hidg0', 'rb+') as _:
                _args_.onraspi = True
        except FileNotFoundError:
            pass

    if _args_.profile:
        yappi.start(builtins=False)

    if _args_.verbose > 0:
        if _args_.procs > 0:
            log.info("Multiproc[{} workers]".format(_args_.procs))
        else:
            log.info("Threaded QMode[{}]".format(_args_.qmode))
        if _args_.filter:
            log.info("Kalman filter[enabled]")
        else:
            log.info("Smoothness[{}]".format(_args_.smoothness))
        log.info("Xgain[{:.2f}] Ygain[{:.2f}]".format(_args_.xgain, _args_.ygain))
        log.info("loading detector, predictor: ", end="")

    cwd = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))
    # model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_5_face_landmarks.dat"))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    if _args_.verbose > 0:
        log.info("done.", popend=True)

    renice(-10, mp.current_process().pid)

    origsig = [signal.signal(signal.SIGINT, sig_handler),
               signal.signal(signal.SIGTERM, sig_handler),
               signal.signal(signal.SIGPIPE, sig_handler), ]
    try:
        if _args_.procs > 0:
            start_face_detect_procs(detector, predictor)
        else:
            start_face_detect_thread(detector, predictor)
    except Exception:
        signal.signal(signal.SIGINT, origsig[0])
        signal.signal(signal.SIGTERM, origsig[1])
        signal.signal(signal.SIGPIPE, origsig[2])
        raise
    if _args_.profile:
        yappi.stop()

        # retrieve thread stats by their thread id (given by yappi)
        threads = yappi.get_thread_stats()
        threads.print_all()
        for thread in threads:
            log.info(
                "Function stats for (%s) (%d)" % (thread.name, thread.id)
            )  # it is the Thread.__class__.__name__
            yappi.get_func_stats(ctx_id=thread.id).print_all()

    # sys.stdout.flush()
    # sys.stderr.flush()

    if restart:
        if _args_.verbose > 0:
            log.info("Auto restarting...\n")
        os.execv(__file__, sys.argv)

    return 0


if __name__ == '__main__':
    log = MyLogger()
    running = True
    restart = False
    global _args_
    global _fps_

    sys.exit(main())
