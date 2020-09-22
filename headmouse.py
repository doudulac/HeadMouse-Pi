#!/usr/bin/env python3

import argparse
import math
import multiprocessing as mp
import os
from os.path import getmtime
import queue
import shlex
import signal
import struct
import subprocess
import sys
from threading import Thread
import traceback

import cv2
import dlib
import yappi
from imutils import face_utils
from imutils.video import FPS
from filterpy.kalman import KalmanFilter
from filterpy.common.discretization import Q_discrete_white_noise
from scipy.linalg import block_diag
import numpy as np


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
        # keep looping infinitely until the thread is stopped
        framenum = 0
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            frame = f.array
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
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                if _args_.verbose > 0:
                    print("Frames captured:", framenum)
                    try:
                        print("Frames orphaned:", self.frame_q.qsize())
                    except (NotImplementedError, AttributeError):
                        pass
                    print("Frames skipped:", self.skipped)
                return

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
                    print("Frames captured:", framenum)
                    try:
                        print("Frames orphaned:", self.frame_q.qsize())
                    except (NotImplementedError, AttributeError):
                        pass
                    print("Frames skipped:", self.skipped)

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


class MouthOpen(object):
    def __init__(self):
        self._vdists = None
        self._cur_vdist = None
        self._hdists = None
        self._cur_hdist = None
        self._open = False

    def update(self, shapes):
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

        if _args_.verbose > 1:
            print("mouth {:.02f} {:.02f} {:.02f} {:.02f} {:.02f}".format(
                vpast, self._cur_vdist, hpast, self._cur_hdist, _r))

    @property
    def open(self):
        return self._open

    def button_up(self):
        return self._open is False

    def button_down(self):
        return self._open is True


class Eyebrows(object):
    def __init__(self, threshold, sticky=False):
        self.threshold = threshold

        self._ebds = None
        self._angs = None
        self._cur_height = None
        self._raised = False
        self._raised_count = 0
        self.sticky = sticky
        self._sticky_raised = False

    def update(self, shapes):
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
        sep = feature_center(shapes[31:36])
        cen = feature_center(shapes)
        angle = cen[1] - sep[1]
        ebc = feature_center([shapes[19], shapes[24]])
        eyec = feature_center(shapes[36:48])
        ebd = point_distance(ebc, eyec)
        _r = ebd / pdist

        _n = 6
        _s = -2
        if self._ebds is None:
            self._ebds = [ebd, ] * _n
            self._angs = [ebd, ] * _n
        else:
            self._ebds.append(self._ebds.pop(0))
            self._ebds[-1] = ebd
            self._angs.append(self._angs.pop(0))
            self._angs[-1] = angle

        self._cur_height = sum(self._ebds[_s:]) / len(self._ebds[_s:])
        past = sum(self._ebds[:_s]) / len(self._ebds[:_s])
        ang = sum(self._angs[_s:]) / len(self._angs[_s:])
        ang_past = sum(self._angs[:_s]) / len(self._angs[:_s])

        if self.sticky:
            raised = self._cur_height - past > self.threshold and angle - ang_past < 2.0
            if not self._raised:
                self._raised = raised
            else:
                lowered = past - self._cur_height > self.threshold * .60 and ang_past - angle < 2.0
                if not self._sticky_raised or self._raised_count > 0:
                    self._raised = not lowered
                elif raised:
                    self._sticky_raised = False

            if self._raised and not self._sticky_raised:
                self._raised_count += 1
                if self._raised_count > int(round(_fps_.fps() * .5)):
                    self._sticky_raised = True
                    self._raised_count = 0
            else:
                self._raised_count = 0

        else:
            self._raised = self._cur_height - past > self.threshold and angle - ang_past < 2.0

        if _args_.verbose > 1:
            print("brows {:.02f} {:.02f} {:+6.02f} {:+6.02f} {:+6.02f} {:.02f} {:.02f} {:.02f}".format(
                past, self._cur_height, ang_past, ang, angle, _r, pdist, ebd))

    @property
    def cur_height(self):
        return self._cur_height

    @property
    def raised(self):
        return self._raised

    def button_up(self):
        return self._raised is False

    def button_down(self):
        return self._raised is True


class Eyes(object):
    def __init__(self, ear_threshold=None, fps=None):
        self._open = False
        if ear_threshold is None:
            ear_threshold = .15
        self._ear_threshold = ear_threshold
        if fps is None:
            fps = 20
        dt = 1.0 / fps
        self._kf = kalmanfilter_dim2_init(dt=dt, Q=1 ** 2, R=.05)

    def update(self, shapes):
        if shapes is None:
            return

        # reye 36,41
        # leye 42,47

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
            print("eyes {:5.02f} {:5.02f} {:6.03f} {}".format(
                ear, self._kf.x[0][0], self._kf.x[1][0], "O" if self._open else "."
            ))

    @property
    def open(self):
        return self._open

    def button_up(self):
        return self.open

    def button_down(self):
        return not self.button_up()


class Nose(object):
    def __init__(self, use_kalman=False, fps=None):
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
            self._kf = kalmanfilter_dim4_init(dt=dt, Q=2.0, R=2.0)
        else:
            self._kf = None

    def update(self, shapes):
        # jaw 0,16
        # rbrow 17,21
        # lbrow 22,26
        # nose 27,30
        # nostr 31,35
        # reye 36,41
        # leye 42,47
        # moutho 48,59
        # mouthi 60,67
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
            nose = [self._kf.x[0][0], self._kf.x[2][0]]
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
    def __init__(self, button1=None, button2=None, button3=None, pausebtn=None,
                 mindeltathresh=None):
        self._fd = None
        self.open_hidg()
        self._dx = None
        self._dy = None
        self._click = None
        self.btn = {1: {'s': 0, 'f': button1},
                    2: {'s': 0, 'f': button2},
                    3: {'s': 0, 'f': button3}, }
        self.pausebtn = {'s': 0, 'f': pausebtn}
        self._paused = False
        self.cpos = None
        self.track_cpos = self._fd is None
        self.maxheight = None
        self.maxwidth = None
        self.wrap = False
        self.xgain = _args_.xgain if _args_.xgain is not None else 1.0
        self.ygain = _args_.ygain if _args_.ygain is not None else 1.0
        self.wrap = False
        self.mindeltathresh = mindeltathresh if mindeltathresh is not None else 1
        self._smoothness = None
        self._motionweight = None
        self.set_smoothness(_args_.smoothness)
        self._prevdx = 0
        self._prevdy = 0
        self.i_accel = None
        self.accel = [1.0, 1.0, 1.8, 1.9, 2.0,
                      2.0, 2.0, 2.0, 2.1, 2.2,
                      2.3, 2.4, 2.5, 3.0, 4.0,
                      4.1, 4.2, 4.3, 4.4, 4.5,
                      4.6, 4.7, 4.8, 4.9, 5.0,
                      5.1, 5.2, 5.3, 5.4, 5.5, ]
        # self.accel = [1.0]*10 + [2.0]*20

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

    def process_movement(self, nose):
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
                self.cpos = nose.position

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
                click |= 1 << (i-1)
            elif btn['s'] == 0 and btn['f'].button_down():
                if i != 1:
                    btn['s'] = int(round(_fps_.fps() * .4))
                click |= 1 << (i-1)

        self._click = click

        return click

    def process_pause(self):
        btn = self.pausebtn

        if btn['f'] is None:
            return False

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
                if _args_.verbose > 0:
                    print('{}pause'.format('' if self.paused else 'un'))

        if btn['s'] == 0 and btn['f'].button_down():
            btn['s'] = int(round(_fps_.fps() * 1.5))

    def update(self, nose):
        self.process_pause()
        if not self.paused:
            dx, dy = self.process_movement(nose)
            click = self.process_clicks()
            self.send_mouse_relative(click, dx, dy)
        elif self.click != 0:
            self._click = 0
            self.send_mouse_relative(0, 0, 0)

    def send_mouse_relative(self, click, dx, dy):
        if _args_.verbose > 1 and click:
            print('click', click)

        if self._fd is not None:
            report = struct.pack('<2b2h', 2, click, dx, dy)
            self._fd.write(report)

    def set_smoothness(self, value):
        if value is None or value < 0:
            value = 0
        elif value > 8:
            value = 8

        self._smoothness = value
        self._motionweight = math.log10(float(self._smoothness) + 1)

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


def annotate_frame(frame, shapes, nose, brows, mouse):
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

    facec = feature_center(shapes)
    cv2.circle(frame, facec, 4, (128, 0, 255), -1)

    draw_landmarks(frame, shapes, facec)
    # frame = face_utils.visualize_facial_landmarks(frame, shapes, [(0,255,0),]*8)
    return frame


def draw_landmarks(frame, shapes, center):
    if shapes is None:
        return
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
            r = args.scalew / float(w)
            dim = (args.scalew, int(h * r))
            if args.verbose > 0 and mp.current_process().name[-2:] == "-1":
                print("Frame shape: {}\nFrame scaled: {}".format(frame.shape, dim), flush=True)

        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sgframe = cv2.resize(gframe, dim, interpolation=cv2.INTER_AREA)
        faces = detector(sgframe)
        if len(faces) == 0:
            shapes = None
        else:
            face = dlib.rectangle(int(faces[0].left() / r),
                                  int(faces[0].top() / r),
                                  int(faces[0].right() / r),
                                  int(faces[0].bottom() / r))
            shapes = predictor(gframe, face)
            shapes = face_utils.shape_to_np(shapes)
        if args.onraspi and not _args_.debug_video:
            frame = None
        shapesq.put_nowait((framenum, frame, shapes))

    shapesq.cancel_join_thread()
    if args.verbose > 0:
        print(mp.current_process().name, "stopped.", flush=True)
    return 0


def face_detect(demoq, detector, predictor):
    global running
    global restart
    global _args_
    global _fps_

    _fps_ = FPS()
    r = None
    rotate = None
    picam = _args_.onraspi and not _args_.usbcam
    frameq = queue.Queue(5) if _args_.qmode else None
    framerate = 20
    resolution = (1280, 720)
    if _args_.onraspi:
        if not _args_.usbcam:
            # resolution = (320, 240)
            # framerate = 24
            resolution = (640, 480)
            framerate = 20
            rotate = 180

    cam = MyVideoStream(usePiCamera=picam, resolution=resolution, framerate=framerate,
                        frame_q=frameq, rotation=rotate).start()

    if _args_.debug_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter("/tmp/testfile.avi", fourcc, framerate, resolution)
    else:
        writer = None

    mouth = MouthOpen()
    eyes = Eyes(ear_threshold=_args_.ear, fps=cam.fps())
    brows = Eyebrows(_args_.ebd, sticky=_args_.stickyclick)
    nose = Nose(_args_.filter, fps=cam.fps())
    mouse = MousePointer(button1=brows, button2=mouth, pausebtn=eyes)
    if _args_.debug:
        mouse.close_hidg()

    no_face_frames = 0
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
            r = _args_.scalew / float(w)
            dim = (_args_.scalew, int(h * r))
            if _args_.verbose > 0:
                print("Frame shape: {}\nFrame scaled: {}".format(frame.shape, dim))

        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sgframe = cv2.resize(gframe, dim, interpolation=cv2.INTER_AREA)
        faces = detector(sgframe)
        if len(faces) == 0:
            no_face_frames += 1
            if _args_.verbose > 0:
                print(no_face_frames, 'no face', end='\r')
            continue

        face = dlib.rectangle(int(faces[0].left() / r),
                              int(faces[0].top() / r),
                              int(faces[0].right() / r),
                              int(faces[0].bottom() / r))
        shapes = predictor(gframe, face)
        shapes = face_utils.shape_to_np(shapes)

        _fps_.stop()
        if framenum > 5 and framenum % int(2 * _fps_.fps()) == 0:
            if getmtime(__file__) != mtime:
                restart = True
                running = False
                break

        mouth.update(shapes)
        eyes.update(shapes)
        nose.update(shapes)
        brows.update(shapes)
        try:
            mouse.update(nose)
        except BrokenPipeError:
            if _args_.verbose > 0:
                traceback.print_exc()
            restart = True
            running = False
            break

        if _args_.verbose >= 3:
            line = "{:4} ({:8.3f}, {:8.3f}) ({:8.3f}, {:6.3f}) ({:8.3f}, {:6.3f}) ".format(
                framenum, nose.nose_raw[0], nose.nose_raw[1], nose.position[0], nose.vel[0],
                nose.position[1], nose.vel[1])
            line += "({:3}, {:5.2f}) ({:3}, {:5.2f}) {:2} {}".format(nose.dx, nose.ax, nose.dy,
                                                                     nose.ay, mouse.i_accel,
                                                                     mouse.accel[mouse.i_accel])
            print(line)

        if not _args_.onraspi or writer is not None:
            annotate_frame(frame, shapes, nose, brows, mouse)

        if writer is not None:
            writer.write(frame)

        if not _args_.onraspi:
            demoq.put_nowait(frame)

        _fps_.update()

    if _args_.verbose > 0 and no_face_frames:
        print('')

    if writer is not None:
        writer.release()

    mouse.send_mouse_relative(0, 0, 0)
    mouse.close_hidg()

    _fps_.stop()
    cfps = cam.fps()
    cam.stop()
    cam.join()

    if _args_.verbose > 0:
        print("Elapsed time: {:.1f}s".format(_fps_.elapsed()))
        print("         FPS: {:.3f}/{}".format(_fps_.fps(), cfps))


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
    parser.add_argument("-r", "--procs", default=2, type=int,
                        help="number of procs (default: 2)")
    parser.add_argument("-s", "--smoothness", default=3, type=int, choices=range(1, 9),
                        help="smoothness 1-8 (default: 3)")
    parser.add_argument("-u", "--usbcam", action="store_true",
                        help="Use usb camera instead of PiCamera")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Verbosity")
    parser.add_argument("-w", "--scalew", default=320, type=int,
                        help="scale width (default: 320)")
    parser.add_argument("-x", "--xgain", default=1.0, type=float,
                        help="X gain")
    parser.add_argument("-y", "--ygain", default=1.0, type=float,
                        help="Y gain")
    parser.add_argument("--ear", default=.15,
                        help="eye aspect ratio threshold (default: .15)")
    parser.add_argument("--onraspi", action="store_true",
                        help="force raspi mode")
    parser.add_argument("--debug", action="store_true",
                        help="disable mouse reports to host")
    parser.add_argument("--debug-video", action="store_true",
                        help="save video to testfile")

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
        print("Adjusting process priority.")
        stdout = None
        stderr = None
    else:
        stdout = subprocess.DEVNULL
        stderr = stdout

    pstr = " ".join(["-p {}".format(p) for p in _pids])
    cmd = shlex.split("/usr/bin/sudo /usr/bin/renice {} {}".format(nice, pstr))
    subprocess.Popen(cmd, stdout=stdout, stderr=stderr).wait(timeout=5)


def start_face_detect_procs(detector, predictor):
    global _fps_
    global restart

    if _args_.verbose > 0:
        print("{} pid {}".format(mp.current_process().name, mp.current_process().pid))
    frameq = mp.Queue(5)
    shapesqs = [mp.Queue()] * _args_.procs
    workers = []
    for i in range(_args_.procs):
        _p = mp.Process(target=face_detect_mp, args=(frameq, shapesqs[i], detector, predictor,
                                                     _args_))
        _p.start()
        workers.append(_p)
        if _args_.verbose > 0:
            print("{} pid {}".format(_p.name, _p.pid))

    renice(-10, [p.pid for p in workers])

    _fps_ = FPS()
    rotate = None
    picam = _args_.onraspi and not _args_.usbcam
    framerate = 20
    resolution = (1280, 720)
    if _args_.onraspi:
        if not _args_.usbcam:
            # resolution = (320, 240)
            # framerate = 24
            resolution = (640, 480)
            framerate = 20
            rotate = 180

    cam = MyVideoStream(usePiCamera=picam, resolution=resolution, framerate=framerate,
                        frame_q=frameq, rotation=rotate).start()

    if _args_.debug_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter("/tmp/testfile.avi", fourcc, framerate, resolution)
    else:
        writer = None

    mouth = MouthOpen()
    eyes = Eyes(ear_threshold=_args_.ear, fps=cam.fps())
    brows = Eyebrows(_args_.ebd, sticky=_args_.stickyclick)
    nose = Nose(_args_.filter, fps=cam.fps())
    mouse = MousePointer(button1=brows, button2=mouth, pausebtn=eyes)
    if _args_.debug:
        mouse.close_hidg()

    firstframe = True
    timeout = None
    mtime = getmtime(__file__)
    ooobuf = {}
    nextframe = 1
    qnum = -1
    try:
        while True:
            try:
                framenum, frame, shapes = ooobuf[nextframe]
            except KeyError:
                try:
                    qnum += 1
                    if qnum >= len(shapesqs):
                        qnum = 0
                    framenum, frame, shapes = shapesqs[qnum].get(timeout=timeout)
                except queue.Empty:
                    if _args_.verbose > 0:
                        print("queue delay...low voltage?")
                    continue
            if framenum != nextframe:
                ooobuf[framenum] = (framenum, frame, shapes)
                continue
            nextframe += 1

            if firstframe:
                firstframe = False
                timeout = 2 / framerate
                mouse.maxwidth, mouse.maxheight = cam.framew, cam.frameh
                _fps_.start()

            _fps_.stop()
            if framenum > 5 and framenum % int(2 * _fps_.fps()) == 0:
                if getmtime(__file__) != mtime:
                    restart = True
                    break

            mouth.update(shapes)
            eyes.update(shapes)
            brows.update(shapes)
            nose.update(shapes)
            mouse.update(nose)

            if _args_.verbose >= 3:
                line = "{:4} ({:8.3f}, {:8.3f}) ({:8.3f}, {:6.3f}) ({:8.3f}, {:6.3f}) ".format(
                    framenum, nose.nose_raw[0], nose.nose_raw[1], nose.position[0], nose.vel[0],
                    nose.position[1], nose.vel[1])
                line += "({:3}, {:5.2f}) ({:3}, {:5.2f}) {:2} {}".format(nose.dx, nose.ax, nose.dy,
                                                                         nose.ay, mouse.i_accel,
                                                                         mouse.accel[mouse.i_accel])
                print(line)

            if not _args_.onraspi or writer is not None:
                annotate_frame(frame, shapes, nose, brows, mouse)

            if writer is not None:
                writer.write(frame)

            if not _args_.onraspi:
                cv2.imshow("Demo", frame)

                if cv2.waitKey(1) == 27:
                    break

            _fps_.update()

    except KeyboardInterrupt:
        pass

    except BrokenPipeError:
        if _args_.verbose > 0:
            traceback.print_exc()
        restart = True

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
            print("Joining {}".format(p.name))
        p.join()

    if _args_.verbose > 0:
        print("Elapsed time: {:.1f}s".format(_fps_.elapsed()))
        print("         FPS: {:.3f}/{}".format(_fps_.fps(), cfps))


def start_face_detect_thread(detector, predictor):
    global running

    fps = FPS()
    demoq = queue.Queue()
    t = Thread(target=face_detect, args=(demoq, detector, predictor))
    t.start()
    firstframe = True
    try:
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
    except KeyboardInterrupt:
        running = False

    fps.stop()
    t.join()

    if _args_.verbose > 0 and not _args_.onraspi:
        print("Demo Queue")
        print("Elapsed time: {:.1f}s".format(fps.elapsed()))
        print("         FPS: {:.3f}".format(fps.fps()))


def main():
    global running
    global restart
    global _args_

    _args_ = parse_arguments()

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
            print("Multiproc[{} workers]".format(_args_.procs))
        else:
            print("Threaded QMode[{}]".format(_args_.qmode))
        if _args_.filter:
            print("Kalman filter[enabled]")
        else:
            print("Smoothness[{}]".format(_args_.smoothness))
        print("Xgain[{:.2f}] Ygain[{:.2f}]".format(_args_.xgain, _args_.ygain))
        print("loading detector, predictor: ", end="", flush=True)
    cwd = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))
    # model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_5_face_landmarks.dat"))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    if _args_.verbose > 0:
        print("done.")

    renice(-10, mp.current_process().pid)

    if _args_.procs > 0:
        start_face_detect_procs(detector, predictor)
    else:
        start_face_detect_thread(detector, predictor)

    if _args_.profile:
        yappi.stop()

        # retrieve thread stats by their thread id (given by yappi)
        threads = yappi.get_thread_stats()
        threads.print_all()
        for thread in threads:
            print(
                "Function stats for (%s) (%d)" % (thread.name, thread.id)
            )  # it is the Thread.__class__.__name__
            yappi.get_func_stats(ctx_id=thread.id).print_all()

    sys.stdout.flush()
    sys.stderr.flush()

    if restart:
        if _args_.verbose > 0:
            print("\nAuto restarting {}\n".format(' '.join(sys.argv)), flush=True)
        os.execv(__file__, sys.argv)

    return 0


if __name__ == '__main__':
    running = True
    restart = False
    global _args_
    global _fps_

    sys.exit(main())
