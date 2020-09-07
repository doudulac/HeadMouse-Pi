#!/usr/bin/env python3

import argparse
import math
import multiprocessing as mp
import os
import queue
import shlex
import signal
import struct
import subprocess
import sys
from queue import Empty
from threading import Thread

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
    def __init__(self, resolution=(320, 240), framerate=32, frame_q=None, **kwargs):
        self.frame_q = frame_q

        # initialize the camera
        import picamera.array

        self.camera = picamera.PiCamera()

        # set camera parameters
        self.camera.resolution = resolution
        self.camera.framerate = framerate

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
                self.frame_q.put_nowait((framenum, frame))
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
                return

    def read(self):
        # return the frame most recently read
        if self.frame_q is not None:
            try:
                (self.framenum, self.frame) = self.frame_q.get(timeout=1)
            except Empty:
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
                return

            # otherwise, read the next frame from the stream
            (_, frame) = self.stream.read()
            if self.framew is None and frame is not None:
                (self.frameh, self.framew) = frame.shape[:2]
            framenum += 1
            if self.frame_q is not None:
                self.frame_q.put_nowait((framenum, frame))
            else:
                self.framenum = framenum
                self.frame = frame

    def read(self):
        # return the frame most recently read
        if self.frame_q is not None:
            try:
                (self.framenum, self.frame) = self.frame_q.get(timeout=1)
            except Empty:
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


class Eyebrows(object):
    def __init__(self, threshold):
        self.threshold = threshold

        self._ebds = None
        self._pos_down = None
        self._position = None
        self._raised = False

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

        ebc = feature_center(shapes[17:27])
        eyec = feature_center(shapes[36:48])
        ebd = point_distance(ebc, eyec)
        if self._ebds is None:
            self._ebds = [ebd, ebd]
        else:
            self._ebds.append(self._ebds.pop(0))
            self._ebds[-1] = ebd
            if self._pos_down is None:
                self._pos_down = sum(self._ebds) / len(self._ebds)

        self._position = sum(self._ebds) / len(self._ebds)
        if self._pos_down is not None:
            self._raised = self._position - self._pos_down > self.threshold

    @property
    def position(self):
        return self._position

    @property
    def pos_down(self):
        return self._pos_down

    @property
    def raised(self):
        return self._raised


class Nose(object):
    def __init__(self, use_kalman=False, kalman_dt=None):
        self._positions = None
        self.nose_raw = [0, 0]
        self._dx = None
        self._dy = None
        self._vels = None
        self._ax = None
        self._ay = None
        if use_kalman:
            self._kf = kalmanfilter_init(kalman_dt)
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
    def __init__(self, xgain=None, ygain=None, smoothness=None, mindeltathresh=None, verbose=None):
        self._dx = None
        self._dy = None
        self.cpos = None
        self.maxheight = None
        self.maxwidth = None
        self.wrap = False
        self.xgain = xgain if xgain is not None else 1.0
        self.ygain = ygain if ygain is not None else 1.0
        self.wrap = False
        self.mindeltathresh = mindeltathresh if mindeltathresh is not None else 1
        self._smoothness = None
        self._motionweight = None
        self.set_smoothness(smoothness)
        self._prevdx = 0
        self._prevdy = 0
        self.verbose = verbose if verbose is not None else 0
        self._fd = None
        self.open_hidg()
        self.i_accel = None
        self.accel = [1.0, 1.0, 1.8, 1.9, 2.0,
                      2.0, 2.0, 2.0, 2.1, 2.2,
                      2.3, 2.4, 2.5, 3.0, 4.0,
                      4.1, 4.2, 4.3, 4.4, 4.5,
                      4.6, 4.7, 4.8, 4.9, 5.0,
                      5.1, 5.2, 5.3, 5.4, 5.5, ]

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

    def update(self, nose, brows):
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

        if brows.raised:
            click = 1
        else:
            click = 0

        self.send(click, dx, dy)
        if self._fd is not None:
            return

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

    def send(self, click, dx, dy):
        if self.verbose >= 3 and click:
            print('click')

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
    def smoothness(self):
        return self._smoothness

    @property
    def motionweight(self):
        return self._motionweight


def annotate_frame(frame, shapes, nose, brows, mouse):
    cv2.circle(frame, (int(mouse.cpos[0]), int(mouse.cpos[1])), 4, (0, 0, 255), -1)

    try:
        _d = brows.position - brows.pos_down
        _pd = brows.pos_down
    except TypeError:
        _d = 0
        _pd = 0
    cv2.putText(frame, "{:.02f}, {}, {:.02f}".format(_pd, brows.raised, _d), (90, 130),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "brows: {:.2f}".format(brows.position), (90, 165),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "nose: " + str(nose.position), (90, 200),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "dxdy: " + str((mouse.dx, mouse.dy)), (90, 235),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "ptr : " + str((int(mouse.cpos[0]), int(mouse.cpos[1]))), (90, 270),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    facec = feature_center(shapes)
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
            if _l == 1:
                cv2.line(frame, center, p1, (0, 255, 255))
            cv2.line(frame, center, p2, (0, 255, 255))


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
                print("Frame shape: {}\nFrame scaled: {}".format(frame.shape, dim))

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
        if args.onraspi:
            frame = None
        shapesq.put_nowait((framenum, frame, shapes))

    shapesq.cancel_join_thread()
    if args.verbose > 0:
        print(mp.current_process().name, "stopped.")
    return 0


def face_detect(demoq, detector, predictor):
    global running
    global _args_

    r = None
    rotate = None
    picam = False
    frameq = queue.Queue() if _args_.qmode else None
    fps = FPS()
    if _args_.onraspi:
        resolution = (640, 480)
        if not _args_.usbcam:
            rotate = 180
            picam = True
    else:
        resolution = (1280, 720)

    cam = MyVideoStream(usePiCamera=picam, resolution=resolution, frame_q=frameq,
                        rotation=rotate).start()

    no_face_frames = 0
    dim = None
    framenum = 0
    brows = Eyebrows(_args_.ebd)
    nose = Nose(_args_.filter, 1 / cam.fps())
    mouse = MousePointer(xgain=_args_.xgain, ygain=_args_.ygain, smoothness=_args_.smoothness,
                         mindeltathresh=1, verbose=_args_.verbose)
    firstframe = True
    while running:
        pframenum = framenum
        (framenum, frame) = cam.read()
        if frame is None or framenum == pframenum:
            continue

        if firstframe:
            firstframe = False
            fps.start()
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

        nose.update(shapes)
        brows.update(shapes)
        mouse.update(nose, brows)

        if _args_.verbose >= 3:
            line = "{:4} ({:8.3f}, {:8.3f}) ({:8.3f}, {:6.3f}) ({:8.3f}, {:6.3f}) ".format(
                framenum, nose.nose_raw[0], nose.nose_raw[1], nose.position[0], nose.vel[0],
                nose.position[1], nose.vel[1])
            line += "({:3}, {:5.2f}) ({:3}, {:5.2f}) {:2} {}".format(nose.dx, nose.ax, nose.dy,
                                                                     nose.ay, mouse.i_accel,
                                                                     mouse.accel[mouse.i_accel])
            print(line)

        if _args_.onraspi:
            fps.update()
            continue

        annotate_frame(frame, shapes, nose, brows, mouse)

        demoq.put_nowait(frame)
        fps.update()

    if _args_.verbose > 0 and no_face_frames:
        print('')

    mouse.send(0, 0, 0)
    mouse.close_hidg()

    fps.stop()
    cfps = cam.fps()
    cam.stop()
    cam.join()

    if _args_.verbose > 0:
        print("Elapsed time: {:.1f}s".format(fps.elapsed()))
        print("         FPS: {:.3f}/{}".format(fps.fps(), cfps))


def feature_center(shapes):
    centx, centy = 0, 0
    if shapes is not None:
        for x, y in shapes:
            centx += x
            centy += y
        centx = int(centx / len(shapes))
        centy = int(centy / len(shapes))
    return centx, centy


def kalmanfilter_init(dt):
    if dt is None:
        dt = 1 / 20
    f = KalmanFilter(dim_x=4, dim_z=2)
    # State Transition matrix
    f.F = np.array([[1., dt, 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., dt],
                    [0., 0., 0., 1.]])
    # Process noise matrix
    q = Q_discrete_white_noise(dim=2, dt=dt, var=5)
    f.Q = block_diag(q, q)
    # Measurement function
    f.H = np.array([[1., 0., 0., 0.],
                    [0., 0., 1., 0.]])
    # Measurement noise matrix
    f.R = np.array([[5., 0.],
                    [0., 5.]])
    # Current state estimate
    f.x = np.array([[0., 0., 0., 0.]]).T
    # Current state covariance matrix
    f.P = np.eye(4) * 1000.
    return f


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--ebd", default=7.0, type=float,
                        help="Eyebrow distance for click (default: 7.0)")
    parser.add_argument("-f", "--filter", action="store_true",
                        help="enable filter")
    parser.add_argument("-p", "--profile", action="store_true",
                        help="enable profiling")
    parser.add_argument("-q", "--qmode", action="store_true",
                        help="enable queue mode")
    parser.add_argument("-r", "--procs", default=2, type=int,
                        help="number of procs (default: 2)")
    parser.add_argument("-s", "--smoothness", default=1, type=int, choices=range(1, 9),
                        help="smoothness 1-8 (default: 1)")
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
    parser.add_argument("--onraspi", action="store_true",
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
    if _args_.verbose > 0:
        print("{} pid {}".format(mp.current_process().name, mp.current_process().pid))
    frameq = mp.Queue()
    shapesq = mp.Queue()
    workers = []
    for _ in range(_args_.procs):
        _p = mp.Process(target=face_detect_mp, args=(frameq, shapesq, detector, predictor, _args_))
        _p.start()
        workers.append(_p)
        if _args_.verbose > 0:
            print("{} pid {}".format(_p.name, _p.pid))

    renice(-10, [p.pid for p in workers])

    rotate = None
    picam = False
    if _args_.onraspi:
        resolution = (640, 480)
        if not _args_.usbcam:
            rotate = 180
            picam = True
    else:
        resolution = (1280, 720)

    cam = MyVideoStream(usePiCamera=picam, resolution=resolution, frame_q=frameq,
                        rotation=rotate).start()

    brows = Eyebrows(_args_.ebd)
    nose = Nose(_args_.filter, 1 / cam.fps())
    mouse = MousePointer(xgain=_args_.xgain, ygain=_args_.ygain, smoothness=_args_.smoothness,
                         mindeltathresh=1, verbose=_args_.verbose)
    fps = FPS()
    firstframe = True
    timeout = None
    try:
        while True:
            try:
                framenum, frame, shapes = shapesq.get(timeout=timeout)
            except queue.Empty:
                if _args_.verbose > 0:
                    print("queue delay...low voltage?")
                continue

            if firstframe:
                firstframe = False
                timeout = .1
                mouse.maxwidth, mouse.maxheight = cam.framew, cam.frameh
                fps.start()

            brows.update(shapes)
            nose.update(shapes)
            mouse.update(nose, brows)

            if _args_.verbose >= 3:
                line = "{:4} ({:8.3f}, {:8.3f}) ({:8.3f}, {:6.3f}) ({:8.3f}, {:6.3f}) ".format(
                    framenum, nose.nose_raw[0], nose.nose_raw[1], nose.position[0], nose.vel[0],
                    nose.position[1], nose.vel[1])
                line += "({:3}, {:5.2f}) ({:3}, {:5.2f}) {:2} {}".format(nose.dx, nose.ax, nose.dy,
                                                                         nose.ay, mouse.i_accel,
                                                                         mouse.accel[mouse.i_accel])
                print(line)

            if _args_.onraspi:
                fps.update()
                continue

            annotate_frame(frame, shapes, nose, brows, mouse)

            cv2.imshow("Demo", frame)
            fps.update()

            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        pass

    mouse.send(0, 0, 0)
    mouse.close_hidg()

    fps.stop()
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
        print("Elapsed time: {:.1f}s".format(fps.elapsed()))
        print("         FPS: {:.3f}/{}".format(fps.fps(), cfps))


def start_face_detect_thread(detector, predictor):
    global running

    fps = FPS()
    demoq = queue.Queue()
    t = Thread(target=face_detect, args=(demoq, detector, predictor))
    t.start()
    firstframe = True
    try:
        while running:
            frame = demoq.get()
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
    global _args_

    _args_ = parse_arguments()

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

    return 0


if __name__ == '__main__':
    running = True
    global _args_

    sys.exit(main())
