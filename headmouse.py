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
import time
from queue import Empty
from threading import Thread

import cv2
import dlib
import yappi
from imutils import face_utils
from imutils.video import FPS
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
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
                    except NotImplementedError:
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


def draw_landmarks(frame, shapes, center):
    for (i, j) in face_utils.FACIAL_LANDMARKS_68_IDXS.values():
        pts = shapes[i:j]
        for _l in range(1, len(pts)):
            p1 = tuple(pts[_l - 1])
            p2 = tuple(pts[_l])
            cv2.line(frame, p1, p2, (0, 255, 0))
            if _l == 1:
                cv2.line(frame, center, p1, (0, 255, 255))
            cv2.line(frame, center, p2, (0, 255, 255))


def feature_center(shapes):
    centx, centy = 0, 0
    for x, y in shapes:
        centx += x
        centy += y
    centx = int(centx / len(shapes))
    centy = int(centy / len(shapes))
    return centx, centy


def point_distance(p1, p2):
    d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return d


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
            if args.verbose > 0 and mp.current_process().name[-2:] == "-1":
                print("Frame shape:", frame.shape)
            (h, w) = frame.shape[:2]
            r = args.scalew / float(w)
            dim = (args.scalew, int(h * r))

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

    ebds = None
    prevdx = 0
    prevdy = 0
    pos = None
    cpos = None
    eb_down = None
    r = None
    maxwidth, maxheight = None, None

    wrap = False
    mindeltathresh = 1
    smoothness = _args_.smoothness  # <= 8
    motionweight = math.log10(float(smoothness) + 1)
    accel = [1.0, 1.0, 1.5, 1.6, 1.7,
             1.8, 1.9, 2.0, 2.1, 2.2,
             2.3, 2.4, 2.5, 3.0, 4.0,
             4.1, 4.2, 4.3, 4.4, 4.5,
             4.6, 4.7, 4.8, 4.9, 5.0,
             5.1, 5.2, 5.3, 5.4, 5.5, ]

    if _args_.onraspi:
        if _args_.usbcam:
            rotate = None
            resolution = (640, 480)
        else:
            rotate = 180
            resolution = (640, 480)
        webcam = MyVideoStream(usePiCamera=not _args_.usbcam, resolution=resolution,
                               qmode=_args_.qmode, rotation=rotate).start()
        time.sleep(2)
        fps = FPS()
        fps.start()
    else:
        webcam = MyVideoStream(src=0, resolution=(1280, 720), qmode=_args_.qmode).start()
        fps = None

    x = 0
    dim = None
    framenum = 0
    while running:
        pframenum = framenum
        (framenum, frame) = webcam.read()
        if frame is None or framenum == pframenum:
            continue

        if dim is None:
            (h, w) = frame.shape[:2]
            maxheight, maxwidth = (h, w)
            r = _args_.scalew / float(w)
            dim = (_args_.scalew, int(h * r))
            # widthratio = 1440 / w
            # heightratio = 900 / h
            # xgain *= widthratio
            # ygain *= heightratio
            # print(frame.shape, dim, (widthratio, heightratio))
            print(frame.shape, dim)

        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sgframe = cv2.resize(gframe, dim, interpolation=cv2.INTER_AREA)
        faces = detector(sgframe)
        if len(faces) == 0:
            x += 1
            print(x, 'no face', end='\r')
            continue

        face = dlib.rectangle(int(faces[0].left() / r),
                              int(faces[0].top() / r),
                              int(faces[0].right() / r),
                              int(faces[0].bottom() / r))
        shapes = predictor(gframe, face)
        shapes = face_utils.shape_to_np(shapes)

        # jaw 0,16
        # rbrow 17,21
        # lbrow 22,26
        # nose 27,30
        # nostr 31,35
        # reye 36,41
        # leye 42,47
        # moutho 48,59
        # mouthi 60,67

        nose = shapes[30]
        facec = feature_center(shapes)
        ebc = feature_center(shapes[17:27])
        eyec = feature_center(shapes[36:48])
        ebd = point_distance(ebc, eyec)

        if pos is None:
            pos = [nose, nose]
            cpos = nose
            ebds = [ebd, ebd]
        else:
            pos.append(pos.pop(0))
            pos[-1] = nose
            ebds.append(ebds.pop(0))
            ebds[-1] = ebd
            if eb_down is None:
                eb_down = sum(ebds) / len(ebds)

        ebda = sum(ebds) / len(ebds)
        if eb_down is not None and ebda - eb_down > _args_.ebd:
            ebr = True
        else:
            ebr = False

        dx = pos[1][0] - pos[0][0]
        dy = pos[1][1] - pos[0][1]

        dx *= _args_.xgain
        dy *= _args_.ygain
        dx = dx * (1.0 - motionweight) + prevdx * motionweight
        dy = dy * (1.0 - motionweight) + prevdy * motionweight
        prevdx = dx
        prevdy = dy

        dist = math.sqrt(dx * dx + dy * dy)
        i_accel = int(dist + 0.5)
        if i_accel >= len(accel):
            i_accel = len(accel) - 1

        if -mindeltathresh < dx < mindeltathresh:
            dx = 0
        if -mindeltathresh < dy < mindeltathresh:
            dy = 0
        # print(i_accel)
        dx *= accel[i_accel]
        dy *= accel[i_accel]
        dx = -int(round(dx))
        dy = int(round(dy))

        if _args_.onraspi:
            if ebr:
                click = 1
                # print('click')
            else:
                click = 0
            report = struct.pack('<2b2h', 2, click, dx, dy)
            with open('/dev/hidg0', 'rb+') as fd:
                fd.write(report)
            fps.update()
            continue

        # On dev system, draw stuff and simulate pointer
        cpos[0] += dx
        cpos[1] += dy

        if cpos[0] > maxwidth:
            if wrap:
                cpos[0] -= maxwidth
            else:
                cpos[0] = maxwidth
        if cpos[1] > maxheight:
            if wrap:
                cpos[1] -= maxheight
            else:
                cpos[1] = maxheight
        if cpos[0] < 0:
            if wrap:
                cpos[0] += maxwidth
            else:
                cpos[0] = 0
        if cpos[1] < 0:
            if wrap:
                cpos[1] += maxheight
            else:
                cpos[1] = 0

        cv2.circle(frame, (int(cpos[0]), int(cpos[1])), 4, (0, 0, 255), -1)

        cv2.putText(frame, str((ebd, ebda, eb_down, ebr)), (90, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # cv2.putText(frame, "brows: " + str(t2), (90, 165),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "nose: " + str(nose), (90, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "dxdy: " + str((dx, dy)), (90, 235),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "ptr : " + str((int(cpos[0]), int(cpos[1]))), (90, 270),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        draw_landmarks(frame, shapes, facec)
        # frame = face_utils.visualize_facial_landmarks(frame, shapes, [(0,255,0),]*8)
        demoq.put(frame)

    if _args_.onraspi:
        report = struct.pack('<2b2h', 2, 0, 0, 0)
        with open('/dev/hidg0', 'rb+') as fd:
            fd.write(report)

    webcam.stop()
    print('')
    if fps is not None:
        fps.stop()
        print(fps.elapsed(), fps.fps())


def start_face_detect_procs(detector, predictor):
    ebds = None
    prevdx = 0
    prevdy = 0
    pos = None
    cpos = None
    eb_down = None
    maxwidth, maxheight = None, None

    if _args_.filter:
        nose_flt = kalmanfilter_init()
    else:
        nose_flt = None

    wrap = False
    mindeltathresh = 1
    smoothness = _args_.smoothness  # <= 8
    motionweight = math.log10(float(smoothness) + 1)
    accel = [1.0, 1.0, 1.5, 1.6, 1.7,
             1.8, 1.9, 2.0, 2.1, 2.2,
             2.3, 2.4, 2.5, 3.0, 4.0,
             4.1, 4.2, 4.3, 4.4, 4.5,
             4.6, 4.7, 4.8, 4.9, 5.0,
             5.1, 5.2, 5.3, 5.4, 5.5, ]

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
    fd = None
    if _args_.onraspi:
        fd = open('/dev/hidg0', 'rb+')
        resolution = (640, 480)
        if not _args_.usbcam:
            rotate = 180
            picam = True
    else:
        resolution = (1280, 720)

    cam = MyVideoStream(usePiCamera=picam, resolution=resolution, frame_q=frameq,
                        rotation=rotate).start()
    fps = FPS()
    firstframe = True
    try:
        while True:
            framenum, frame, shapes = shapesq.get()
            if firstframe:
                firstframe = False
                fps.start()
            if shapes is None:
                fps.update()
                continue
            nose = shapes[30]

            if nose_flt is not None:
                nose_flt.predict()
                nose_flt.update(nose)
                nose = [nose_flt.x[0][0], nose_flt.x[2][0]]
                nose_v = [nose_flt.x[1][0], nose_flt.x[3][0]]
            else:
                nose_v = [0, 0]

            facec = feature_center(shapes)
            ebc = feature_center(shapes[17:27])
            eyec = feature_center(shapes[36:48])
            ebd = point_distance(ebc, eyec)

            if pos is None:
                pos = [nose, nose]
                vel = [nose_v, nose_v]
                cpos = nose
                ebds = [ebd, ebd]
                maxwidth, maxheight = cam.framew, cam.frameh
            else:
                pos.append(pos.pop(0))
                pos[-1] = nose
                vel.append(vel.pop(0))
                vel[-1] = nose_v
                ebds.append(ebds.pop(0))
                ebds[-1] = ebd
                if eb_down is None:
                    eb_down = sum(ebds) / len(ebds)

            ebda = sum(ebds) / len(ebds)
            if eb_down is not None and ebda - eb_down > _args_.ebd:
                ebr = True
            else:
                ebr = False

            dx = pos[1][0] - pos[0][0]
            dy = pos[1][1] - pos[0][1]
            # dx = nose_v[0] / 2
            # dy = nose_v[1] / 2
            ax = vel[1][0] - vel[0][0]
            ay = vel[1][1] - vel[0][1]

            dx *= _args_.xgain
            dy *= _args_.ygain
            if not _args_.filter:
                dx = dx * (1.0 - motionweight) + prevdx * motionweight
                dy = dy * (1.0 - motionweight) + prevdy * motionweight
                prevdx = dx
                prevdy = dy

            dist = math.sqrt(dx * dx + dy * dy)
            i_accel = int(dist + 0.5)
            if i_accel >= len(accel):
                i_accel = len(accel) - 1

            if not _args_.filter:
                if -mindeltathresh < dx < mindeltathresh:
                    dx = 0
                if -mindeltathresh < dy < mindeltathresh:
                    dy = 0
            dx *= accel[i_accel]
            dy *= accel[i_accel]
            dx = -int(round(dx))
            dy = int(round(dy))
            if _args_.verbose >= 3:
                print(
                    "{:4} ({:8.3f}, {:8.3f}) ({:8.3f}, {:6.3f}) ({:8.3f}, {:6.3f}) ({:3}, {:5.2f}) ({:3}, {:5.2f}) {:2} {}".format(
                        framenum, shapes[30][0], shapes[30][1], nose[0], nose_v[0], nose[1],
                        nose_v[1], dx, ax, dy, ay, i_accel, accel[i_accel]))

            if ebr:
                click = 1
                if _args_.verbose >= 3:
                    print('click')
            else:
                click = 0

            if fd is not None:
                report = struct.pack('<2b2h', 2, click, dx, dy)
                fd.write(report)
                fd.flush()
                fps.update()
                continue

            # On dev system, draw stuff and simulate pointer
            cpos[0] += dx
            cpos[1] += dy

            if cpos[0] > maxwidth:
                if wrap:
                    cpos[0] -= maxwidth
                else:
                    cpos[0] = maxwidth
            if cpos[1] > maxheight:
                if wrap:
                    cpos[1] -= maxheight
                else:
                    cpos[1] = maxheight
            if cpos[0] < 0:
                if wrap:
                    cpos[0] += maxwidth
                else:
                    cpos[0] = 0
            if cpos[1] < 0:
                if wrap:
                    cpos[1] += maxheight
                else:
                    cpos[1] = 0

            cv2.circle(frame, (int(cpos[0]), int(cpos[1])), 4, (0, 0, 255), -1)

            cv2.putText(frame, str((ebd, ebda, eb_down, ebr)), (90, 130),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            # cv2.putText(frame, "brows: " + str(t2), (90, 165),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "nose: " + str(nose), (90, 200),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "dxdy: " + str((dx, dy)), (90, 235),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "ptr : " + str((int(cpos[0]), int(cpos[1]))), (90, 270),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

            draw_landmarks(frame, shapes, facec)
            # frame = face_utils.visualize_facial_landmarks(frame, shapes, [(0,255,0),]*8)
            cv2.imshow("Demo", frame)
            fps.update()

            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        pass

    if fd is not None:
        report = struct.pack('<2b2h', 2, 0, 0, 0)
        fd.write(report)
        fd.close()

    fps.stop()
    cam.stop()
    cam.join()

    for i in range(_args_.procs):
        frameq.put('STOP')
    for p in workers:
        if _args_.verbose > 0:
            print("Joining {}".format(p.name))
        p.join()

    if _args_.verbose > 0:
        print("Elapsed time: {:.1f}s\n         FPS: {:.3f}".format(fps.elapsed(), fps.fps()))


def start_face_detect_thread(detector, predictor):
    global running

    fps = FPS()
    demoq = queue.Queue()
    t = Thread(target=face_detect, args=(demoq, detector, predictor))
    t.start()
    fps.start()
    try:
        while running:
            frame = demoq.get()

            cv2.imshow("Demo", frame)
            fps.update()

            if cv2.waitKey(1) == 27:
                running = False
    except KeyboardInterrupt:
        running = False

    fps.stop()
    t.join()
    print(fps.elapsed(), fps.fps())


def kalmanfilter_init():
    f = KalmanFilter(dim_x=4, dim_z=2)
    framerate = 20
    dt = 1 / framerate
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


def renice(nice, pids):
    if not _args_.onraspi:
        return

    if isinstance(pids, int):
        _pids = [pids,]
    else:
        _pids = pids

    devnull = ""
    if _args_.verbose:
        print("Adjusting process priority.")
    else:
        devnull = ">/dev/null 2>&1"
    pstr = " ".join(["-p {}".format(p) for p in _pids])
    cmd = shlex.split("/usr/bin/sudo /usr/bin/renice {} {} {}".format(nice, pstr, devnull))
    subprocess.Popen(cmd).wait(timeout=5)


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
        yappi.start(builtins=True)

    if _args_.verbose >= 1:
        print("Xgain: {:.2f}\nYgain: {:.2f}".format(_args_.xgain, _args_.ygain))
        print("loading detector, predictor: ", end="", flush=True)
    cwd = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))
    # model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_5_face_landmarks.dat"))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    if _args_.verbose >= 1:
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
