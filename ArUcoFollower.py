import subprocess
import os
import threading

def play_this(say, delay: float = 0):
    audio_file = "Sounds/" + say + ".mp3"
    null_output = open(os.devnull, 'w')
    fpath = "ffmpeg/bin/ffplay.exe"
    if delay == 0:
        subprocess.Popen([fpath, "-nodisp", "-autoexit", audio_file], stdout=null_output, stderr=null_output)
    else:
        th = threading.Thread(target=delayed_play, args=(say, delay))
        th.start()


def delayed_play(say, delay):
    time.sleep(delay)
    audio_file = "Sounds/" + say + ".mp3"
    null_output = open(os.devnull, 'w')
    fpath = "ffmpeg/bin/ffplay.exe"
    subprocess.Popen([fpath, "-nodisp", "-autoexit", audio_file], stdout=null_output, stderr=null_output)


print("[Program] Importing libraries...")
play_this("importing libraries")

import cv2 as cv
import keyboard
from djitellopy import Tello
import numpy as np
import time
import pygetwindow as gw
import pygame
import pyvjoy
import mss
import tensorflow as tf
from tensorflow import keras
import win32gui
import win32con


play_this("finished importing")
print("[Program] Finished importing libraries.")
transition_time_start: float = time.time()

# CHANGEABLE PARAMETERS-------------------------------------------------------------------------------------------------
vidsource = "dronesim"  # defcam, webcam, obs, tello, dronesim_phone
simmode = "dronesim"  # none, tello, dronesim
_control = "joystick"  # keyboard, joystick
MARKER_SIZE: float = 0.2  # meters (measure your printed marker size)
marker_dist = 2  # meters
training: bool = False
neural_control: bool = True
trainDir = "xr"
comp = "laptop"  # pc, laptop
# -----------------------------------------------------------------------------------------------------------------------
_realtelloflight: bool = True
_automove = False  # only if simmode is tello
key_stop = "esc"
key_ctrlswitch = 'c'
keytakeoffland = 'z'
jsHighS: int = 5  # high speed at R1 of joystick
_fps: float = 10  # fps
_targetAr = 0
_takeofflandwait: int = 3  # in seconds, wait before taking-off / landing

# Parameters prompt
print(f"[Program] Video Source: {vidsource} | Simulation Mode: {simmode} | Real Tello Flight: {_realtelloflight}")
print(f"[Program] Press [{key_stop}, {key_ctrlswitch}, {keytakeoffland}] to [Exit, Toggle AutoFlight, Takeoff/Land")

# INITIALIZATIONS
_mainsleep: float = 1 / _fps
cap = None
tello = Tello() if simmode == "tello" else None
pygame.init() if _control == "joystick" else None
pygame.joystick.init() if _control == "joystick" else None
frame = None
vj = None
freeze_time_limit = 5
jsNeu: int = 16384  # Neutral position (centered)
jsMax: int = 32768  # Maximum position
sct = mss.mss() if vidsource in ["dronesim", "dronesim_phone"] else None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set to '2' to suppress INFO logs in tensorflow
_flying = False
_rl, _fb, _ud, _rrl = 0.0, 0.0, 0.0, 0.0
vj1 = pyvjoy.VJoyDevice(1) if _control == "joystick" else None
frame_time = _mainsleep

# Initialize the frame capturing
if vidsource == "defcam":
    cap = cv.VideoCapture(0)
elif vidsource == "webcam":
    cap = cv.VideoCapture(1)
elif vidsource == "obs":
    cap = cv.VideoCapture(2)  # webcam plugged
    if not cap.isOpened():
        cap = cv.VideoCapture(1)  # no webcam plugged
elif vidsource == "tello":
    # Initialize the Tello drone
    try:  # if tello.ping()
        tello.connect()
    except:
        print("[Program] Can not connect to the drone. Program is aborted")
        exit()
    tello.streamon()

# ARUCO:    Dictionary and Parameters ArUco markers defining
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
parameters = cv.aruco.DetectorParameters()
# Adjusted parameters for lower accuracy but more robust detection
parameters.minMarkerPerimeterRate = 0.01 / (float(MARKER_SIZE / 0.05) * 4)  # Adjust to a smaller value
parameters.maxMarkerPerimeterRate = 20.0 + float(MARKER_SIZE / 0.05) * 10  # Adjust to a larger value
#parameters.polygonalApproxAccuracyRate = 0.3  # Increase the value
#parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_NONE  # Disable corner refinement
#parameters.errorCorrectionRate = 0.1  # Decrease the error correction rate

detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

# POSE ESTIMATE: load in the calibration data
script_directory = os.path.dirname(os.path.abspath(__file__))
calib_data_path = script_directory + "/calibration_data/" + vidsource + "/calib_" + vidsource + ".npz"
calib_data = np.load(calib_data_path)  # print(calib_data.files)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

objPoints = np.zeros((4, 1, 3), dtype=np.float32)
objPoints[0][0] = [-MARKER_SIZE / 2.0, MARKER_SIZE / 2.0, 0]
objPoints[1][0] = [MARKER_SIZE / 2.0, MARKER_SIZE / 2.0, 0]
objPoints[2][0] = [MARKER_SIZE / 2.0, -MARKER_SIZE / 2.0, 0]
objPoints[3][0] = [-MARKER_SIZE / 2.0, -MARKER_SIZE / 2.0, 0]


#  MOVEMENT
def _move(mvec):
    _r: float
    _f: float
    _u: float
    _cl: float
    _r, _f, _u, _cl = mvec
    if simmode == "tello" or simmode == "none":
        _speed: int = 100
        if simmode == "tello" and _automove:
            _speed = 30
        rl: int = int(_speed * _r)
        fb: int = int(_speed * _f)
        ud: int = int(_speed * _u)
        rrl: int = int(_speed * _cl)
        if simmode == "tello" and _realtelloflight:
            tello.send_rc_control(rl, fb, ud, rrl)
        elif (simmode == "none" or (simmode == "tello" and not _realtelloflight)) and _automove:
            print(f"[Program] Auto: {rl}, {fb}, {ud}, {rrl}")
        if not _automove:
            print(f"[Program] Manual: {rl}, {fb}, {ud}, {rrl}")
    if simmode == "dronesim":
        if _control == "keyboard":
            keyboard.press("d") if _r == 1 else keyboard.release("d")
            keyboard.press("a") if _r == -1 else keyboard.release("a")
            keyboard.press("w") if _f == 1 else keyboard.release("w")
            keyboard.press("s") if _f == -1 else keyboard.release("s")
            keyboard.press("up") if _u == 1 else keyboard.release("up")
            keyboard.press("down") if _u == -1 else keyboard.release("down")
            keyboard.press("right") if _cl == 1 else keyboard.release("right")
            keyboard.press("left") if _cl == -1 else keyboard.release("left")
        if _control == "joystick":
            _r = np.sign(_r) if np.abs(_r) > 1 else _r
            _f = np.sign(_f) if np.abs(_f) > 1 else _f
            _u = np.sign(_u) if np.abs(_u) > 1 else _u
            _cl = np.sign(_cl) if np.abs(_cl) > 1 else _cl
            prl = jsNeu + _r * jsNeu
            pfb = jsNeu - _f * jsNeu
            pud = jsNeu - _u * jsNeu
            pcl = jsNeu + _cl * jsNeu
            rl: int = int(prl)
            fb: int = int(pfb)
            ud: int = int(pud)
            cl: int = int(pcl)

            vj1.set_axis(pyvjoy.HID_USAGE_Z, rl)  # RL
            vj1.set_axis(pyvjoy.HID_USAGE_RY, fb)  # FB
            vj1.set_axis(pyvjoy.HID_USAGE_Y, ud)  # UD
            vj1.set_axis(pyvjoy.HID_USAGE_X, cl)  # CLCL


# MOVEMENT: Stop movement
def _stop(axis, mvec):
    rl, fb, ud, rrl = mvec
    if axis == "x":
        if simmode == "tello":
            rl = 0
        if simmode == "dronesim":
            if _control == "keyboard":
                keyboard.release("d")
                keyboard.release("a")
            if _control == "joystick":
                vj1.set_axis(pyvjoy.HID_USAGE_Z, jsNeu)
    if axis == "y":
        if simmode == "tello":
            ud = 0
        if simmode == "dronesim":
            if _control == "keyboard":
                keyboard.release("up")
                keyboard.release("down")
            if _control == "joystick":
                vj1.set_axis(pyvjoy.HID_USAGE_Y, jsNeu)
    if axis == "z":
        if simmode == "tello":
            fb = 0
        if simmode == "dronesim":
            if _control == "keyboard":
                keyboard.release("w")
                keyboard.release("s")
            if _control == "joystick":
                vj1.set_axis(pyvjoy.HID_USAGE_RY, jsNeu)
    if axis == "r":
        if simmode == "tello":
            rrl = 0
        if simmode == "dronesim":
            if _control == "keyboard":
                keyboard.release("left")
                keyboard.release("right")
            if _control == "joystick":
                vj1.set_axis(pyvjoy.HID_USAGE_X, jsNeu)
    if simmode == "tello" and _realtelloflight:
        tello.send_rc_control(rl, fb, ud, rrl)
    if simmode == "none" or (simmode == "tello" and not _realtelloflight):
        print(f"[Program] stop ({rl}, {fb}, {ud}, {rrl})")


def imageShow(frame):
    cv.imshow("ArUco Marker Detection", frame)
    cv.resizeWindow("ArUco Marker Detection", frame.shape[1], frame.shape[0])  # Set the window size

    cv.waitKey(int(_mainsleep * 1000))
    cv.destroyAllWindows()


def neural_xr(xp, rp, zp):
    global _rl, _rrl
    input_xr = np.array([[xp, rp, zp]])
    predicted_outputs_y = model_xr.predict(input_xr)
    _rl, _rrl = predicted_outputs_y[0]


def neural_y(yp, zp):
    global _ud
    input_y = np.array([[yp, zp]])
    predicted_outputs_y = model_y.predict(input_y)
    _ud = predicted_outputs_y[0]


def neural_z(zp: float):
    global _fb
    input_z = zp
    predicted_outputs_z = model_z.predict(input_z)
    _fb = predicted_outputs_z[0]


corners = None
ids = None


def detectMarkers(gray, frame):
    global corners, ids
    corners, ids, _ = detector.detectMarkers(gray, frame)


arXc = np.empty(0)
arYc = np.empty(0)
arZc = np.empty(0)
arRc = np.empty(0)
arXp = np.empty(0)
arYp = np.empty(0)
arZp = np.empty(0)
arRp = np.empty(0)
arXct = np.empty(0)
arYct = np.empty(0)
arZct = np.empty(0)
arRct = np.empty(0)
arXpt = np.empty(0)
arYpt = np.empty(0)
arZpt = np.empty(0)
arRpt = np.empty(0)

# SPECIAL: TELLO
if simmode == "tello":
    print(f"[Program] Battery: {tello.get_battery()}")
    print(f"[Program] Temp: {tello.get_temperature()}")
    if tello.get_battery() < 15:
        print("[Program] Battery is at low level for a start.")
        print("[Program] Program aborted.")
        exit()

# SPECIAL: DRONE SIM
if simmode == "dronesim":

    window = gw.getWindowsWithTitle("Drone Simulation")
    if window:
        window[0].activate()
    else:
        play_this("open drone sim", 1)
        print("[Program] Drone Simulation is not yet open. Trying to open the simulation...")
        time.sleep(2.75)
        subprocess.Popen("Drone Simulation - Shortcut.lnk", shell=True)
        _flying = True
        time.sleep(2)
    _mvec = 0, 0, 0, 0
    _stop("x", _mvec)
    _stop("y", _mvec)
    _stop("z", _mvec)
    _stop("r", _mvec)
    time.sleep(2)
    window = gw.getWindowsWithTitle("Drone Simulation")
    if window:
        window[0].activate()
        print("[Program] Drone Simulation is now launching.")
    else:
        print("[Program] Failed to open Drone Simulation. "
              "Please open it first.")
        time.sleep(1.5)
        play_this("closing the program")
        print("[Program] Closing the program...")
        time.sleep(1.5)
        play_this("successfully closed")
        print("[Program] Program was successfully closed.")
        time.sleep(1.5)
        exit()
    # reposition window
    if vidsource == "dronesim_phone":
        window[0].moveTo(1910, 0)

# SPECIAL: DRONESIM: Check if any joysticks are connected
joystick = None
if _control == "joystick":
    if pygame.joystick.get_count() == 0:
        print("[Program] No joysticks detected.")
        print("[Program] Waiting for joystick connection...")
        print(f"[Program] Press {key_stop} to cancel")
        while True:
            time.sleep(0.05)
            if pygame.joystick.get_count() > 1:
                break
            if cv.waitKey(1) & keyboard.is_pressed(key_stop):
                exit()
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

# Neural
model_xr = None
model_y = None
model_z = None

unit_count = 16
act = 'tanh'
# Define a custom model that matches the architecture
models = [(model_xr, "model_xr"), (model_y, "model_y"), (model_z, "model_z")]

for model, model_name in models:
    input_num, output_num = 3, 2  # Default values
    if model_name == "model_y":
        input_num, output_num = 2, 1
    elif model_name == "model_z":
        input_num, output_num = 1, 1

    model = keras.Sequential([
        keras.layers.Input(shape=input_num, name='input_layer'),  # Inputs
        keras.layers.Dense(unit_count, activation=act, name='hidden_layer_1'),
        keras.layers.Dense(unit_count, activation=act, name='hidden_layer_2'),
        keras.layers.Dense(unit_count, activation=act, name='hidden_layer_3'),
        keras.layers.Dense(output_num, activation=act, name='output_layer')  # Outputs
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # Assign the model back to its original variable
    if model_name == "model_xr":
        model_xr = model
    elif model_name == "model_y":
        model_y = model
    elif model_name == "model_z":
        model_z = model


if neural_control:
    model_xr_path = script_directory + "/calibration_data/" + vidsource + "/model_" + vidsource + "_xr.keras"
    model_xr.load_weights(model_xr_path)
    #model_xr = keras.models.load_model(model_xr_path)
    model_y_path = script_directory + "/calibration_data/" + vidsource + "/model_" + vidsource + "_y.keras"
    model_y.load_weights(model_y_path)
    #model_y = keras.models.load_model(model_y_path)
    model_z_path = script_directory + "/calibration_data/" + vidsource + "/model_" + vidsource + "_z.keras"
    model_z.load_weights(model_z_path)
    # = keras.models.load_model(model_z_path)
    full_model_path = script_directory + "/calibration_data/" + vidsource + "/full_model_" + vidsource + ".h5"
    full_model = keras.models.load_model(full_model_path)
    combined_model_path = script_directory + "/calibration_data/" + vidsource + "/combined_model_" + vidsource + ".h5"
    combined_model = keras.models.load_model(combined_model_path)
    combined_model.summary()

# MAIN RUN -------------------------------------------------------------------------------------------------------------
prevTime: float = 0
deltaTime: float = 0
prev_xt: float = 0
prev_yt: float = 0
prev_zt: float = 0
prev_zr: float = 0
_idle = True  # means no aruco is found
_ckeyisup = True
_zkeyisup = True
_rkeyisup = True
_delkeyisup = True
_savekeyisup = True
_appendkeyisup = True
_recording = False
_highS = False
_prerecord = True
_takeofflandcount = 0
_takeofflandcount = int(_takeofflandwait / frame_time) if not (simmode == "tello" and _realtelloflight) else 0
initiated_frame = False
frozen_frame = False
vid_connected = False
old_frame = None
no_sleep_frame_time = 0

if time.time() - transition_time_start > 5:
    play_this("trying video")
freeze_start_time: float = time.time()

while True:

    testtimestart = time.time()
    sleep_time = 0

    #time.sleep(_mainsleep)


    deltaTime = time.time() - prevTime
    prevTime = time.time()

    # Exit the loop if the stop key is pressed
    if cv.waitKey(1) & keyboard.is_pressed(key_stop):
        break

    if joystick.get_button(4) and joystick.get_button(6):
        break

    for event in pygame.event.get():  # Needed it for joystick
        if event.type == pygame.QUIT:
            _ = None

    # Switch Transfer control to manual / automatic
    if keyboard.is_pressed(key_ctrlswitch) or (joystick.get_button(4) and not joystick.get_button(6)):  # avoid @ close
        if _ckeyisup:
            if _flying:
                if _automove:
                    _automove = False
                    _mvec = 0, 0, 0, 0
                    _stop("x", _mvec)
                    _stop("y", _mvec)
                    _stop("z", _mvec)
                    _stop("r", _mvec)
                    play_this("switched to manual") if not training else None
                    print("[Program]: Control is manual.")
                    if training:
                        _recording = False
                        play_this("paused")
                        print("[Program] Paused recording.")
                else:
                    _automove = True
                    play_this("following aruco") if not training else None
                    print("[Program] Control is automatic.")
                    if training:
                        if len(arXct) > 1:
                            arXp = np.append(arXp, arXpt)
                            arXc = np.append(arXc, arXct)
                            arYp = np.append(arYp, arYpt)
                            arYc = np.append(arYc, arYct)
                            arZp = np.append(arZp, arZpt)
                            arZc = np.append(arZc, arZct)
                            arRp = np.append(arRp, arRpt)
                            arRc = np.append(arRc, arRct)
                            arXpt = np.empty(0)
                            arXct = np.empty(0)
                            arYpt = np.empty(0)
                            arYct = np.empty(0)
                            arZpt = np.empty(0)
                            arZct = np.empty(0)
                            arRpt = np.empty(0)
                            arRct = np.empty(0)
                            print(f"[Program] Data size: {len(arXc)}")
                            play_this("appended_recording")
                            print("[Program] Last recording was appended.")
                        else:
                            play_this("recording")
                        _recording = True
                        _prerecord = True
                        print("[Program] Started recording.")
            else:
                play_this("fly first")
                print("[Program] Fly first before switching to automatic control.")
            _ckeyisup = False

    else:
        _ckeyisup = True

    if training:
        if joystick.get_button(0):
            if _appendkeyisup:
                _appendkeyisup = False
                if len(arXct) > 1:
                    if not _recording:
                        arXp = np.append(arXp, arXpt)
                        arXc = np.append(arXc, arXct)
                        arYp = np.append(arYp, arYpt)
                        arYc = np.append(arYc, arYct)
                        arZp = np.append(arZp, arZpt)
                        arZc = np.append(arZc, arZct)
                        arRp = np.append(arRp, arRpt)
                        arRc = np.append(arRc, arRct)
                        print(f"[Program] Added data size: {len(arXct)}")
                        arXpt = np.empty(0)
                        arXct = np.empty(0)
                        arYpt = np.empty(0)
                        arYct = np.empty(0)
                        arZpt = np.empty(0)
                        arZct = np.empty(0)
                        arRpt = np.empty(0)
                        arRct = np.empty(0)

                        play_this("appended")
                        print("[Program] Last recording was appended.")
                    else:
                        print("[Program] Pause the recording first.")
        else:
            _appendkeyisup = True
        if joystick.get_button(1):
            if _delkeyisup:
                if not _recording:
                    _delkeyisup = False
                    arXpt = np.empty(0)
                    arXct = np.empty(0)
                    arYpt = np.empty(0)
                    arYct = np.empty(0)
                    arZpt = np.empty(0)
                    arZct = np.empty(0)
                    arRpt = np.empty(0)
                    arRct = np.empty(0)
                    play_this("deleted")
                    print("[Program]: Last recording was deleted.")
                else:
                    print("[Program] Pause the recording first.")
        else:
            _delkeyisup = True
        if joystick.get_button(2):
            if _savekeyisup:
                _savekeyisup = False
                save_data_path = script_directory + "/calibration_data/" + vidsource + "/training_" + vidsource\
                                + "_" + trainDir + ".npz"
                np.savez(save_data_path, Xp=arXp, Yp=arYp, Zp=arZp, Rp=arRp, Xc=arXc, Yc=arYc, Zc=arZc, Rc=arRc)

                csv_data_path = script_directory + "/calibration_data/" + vidsource + "/training_" + vidsource\
                                + "_" + trainDir + ".csv"
                Xp = arXp
                Yp = arYp
                Zp = arZp
                Rp = arRp
                Xc = arXc
                Yc = arYc
                Zc = arZc
                Rc = arRc
                stacked_data = np.column_stack((Xp, Yp, Zp, Rp, Xc, Yc, Zc, Rc))
                np.savetxt(csv_data_path, stacked_data, delimiter=',', header='Xp, Yp, Zp, Rp, Xc, Yc, Zc, Rc',
                           comments='')
                play_this("NPZ saved")
                print("[Program]: Saved as npz.")
        else:
            _savekeyisup = True

    # ---Manual Movement------------------------------------------------------------------------------------------------
    if (not _automove) and (simmode == "tello" or simmode == "none"):
        if _flying:
            _rl: float = 0
            _fb: float = 0
            _ud: float = 0
            _rrl: float = 0
            if _control == "keyboard":
                if keyboard.is_pressed("w"):
                    _fb = 1
                if keyboard.is_pressed("s"):
                    _fb = -1
                if keyboard.is_pressed("a"):
                    _rl = -1
                if keyboard.is_pressed("d"):
                    _rl = 1
                if keyboard.is_pressed("up"):
                    _ud = 1
                if keyboard.is_pressed("down"):
                    _ud = -1
                if keyboard.is_pressed("left"):
                    _rrl = -1
                if keyboard.is_pressed("right"):
                    _rrl = 1
                shouldstop = True
                for i in ["w", "a", "s", "d", "up", "down", "left", "right"]:
                    if keyboard.is_pressed(i):
                        shouldstop = False
                if shouldstop:
                    _mvec = _rl, _fb, _ud, _rrl
                    _stop("X", _mvec)
                    _stop("Y", _mvec)
                    _stop("Z", _mvec)
                    _stop("R", _mvec)
            if _control == "joystick":
                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        if event.button == jsHighS:
                            _highS = True
                    if event.type == pygame.JOYBUTTONUP:
                        if event.button == jsHighS:
                            _highS = False
                        if event.button == 9:  # start button
                            break  # break while loop

                _rrl = round(joystick.get_axis(0), 5)
                _ud = round(joystick.get_axis(1), 5) * -1
                _rl = round(joystick.get_axis(2), 5)
                _fb = round(joystick.get_axis(4), 5) * -1

            _mvec = _rl, _fb, _ud, _rrl
            _move(_mvec)
    # ---End Manual Movement------------------------------------------

    # Read a frame from the camera FIX THIS source of temporary error
    has_frame: bool = False
    if vidsource == "tello":
        try:
            frame = tello.get_frame_read().frame
            b, g, r = cv.split(frame)  # to solve blue-red interchange error
            frame = cv.merge((r, g, b))
            has_frame = True
        except Exception as e:
            print(f"[Program] Video Error: {str(e)}")
            print("[Program] Waiting for video...")
    elif vidsource in ["dronesim", "dronesim_phone"]:
        try:
            # _mon = sct.monitors[1]
            monitor_dim = None
            # monitor_dim = {"top": 0, "left": 720, "width": 720, "height": 430   } # MANGABAYPC
            if vidsource == "dronesim":
                monitor_dim = {"top": 0, "left": 960, "width": 960, "height": 540}  # Laptop primary
                if comp == "pc":
                    monitor_dim = {"top": 0, "left": 720, "width": 720, "height": 430   } # MANGABAYPC
            elif vidsource == "dronesim_phone":
                monitor_dim = {"top": 40, "left": 1920, "width": int(960), "height": int(720)}

            frame = np.array(sct.grab(monitor_dim))  # Convert the captured screen image to a NumPy array
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            b, g, r = cv.split(frame)  # to solve blue-red interchange error
            frame = cv.merge((r, g, b))
            has_frame = True
        except:
            print("[Program] Waiting for video...")
    else:
        has_frame, frame = cap.read()

    if has_frame:  # successful capture

        if np.array_equal(old_frame, frame):
            if time.time() - freeze_start_time >= freeze_time_limit:
                play_this("video has frozen")
                print("[Program] Video has frozen.")
                break
        else:
            freeze_start_time = time.time()
        old_frame = frame

        if not initiated_frame:
            initiated_frame = True
            play_this("you can now start", 0.5)
            print("[Program] You can now start.")
        if frozen_frame:
            frozen_frame = False
            play_this("video frame resumed")
            print("[Program] Video frame has resumed.")

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # cv.namedWindow("ArUco Marker Detection", cv.WINDOW_NORMAL)  # make it resizable

        # Detect ArUco markers in the frame
        corners, ids, _ = detector.detectMarkers(gray, frame)
        #th_detect = threading.Thread(target=detectMarkers, args=(gray, frame))
        #th_detect.start()

        sleep_time = _mainsleep - no_sleep_frame_time
        if sleep_time < 0:
            sleep_time = 0
        #time.sleep(sleep_time)

        # IF MARKERS ARE DETECTED
        if ids is not None:
            target_inID: int = 0

            cv.aruco.drawDetectedMarkers(frame, corners)

            # Find target ArUco
            _idle = True
            for i in range(len(ids)):
                if ids[i] == _targetAr:
                    target_inID = i
                    _idle = False

            # IF TARGET ARUCO IS FOUND
            if not _idle:

                # New Pose Estimate
                _, rv, tv = cv.solvePnP(objPoints, corners[target_inID], cam_mat, dist_coef, False,
                                        cv.SOLVEPNP_IPPE_SQUARE)
                cv.drawFrameAxes(frame, cam_mat, dist_coef, rv, tv, 0.1)
                cv.imshow("ArUco Marker Detection", frame)

                # print(f"[Program] Marker {ids[target_inID]} Location: X={tv[0]}, Y={tv[1]}, Z={tv[2]}")

                # ---AUTO MOVEMENT ---
                if _automove and _flying:
                    xt = tv[0]
                    yt = tv[1]
                    zt = tv[2]
                    zr = rv[2]

                    lat: float = 0.0  # sec
                    base_delay: float = 0  # sec
                    delay_factor = base_delay + lat * 2.6

                    xt = xt + delay_factor * (xt - prev_xt)
                    yt = yt + delay_factor * (yt - prev_yt)
                    zt = zt + delay_factor * (zt - prev_zt)
                    zr = zr + delay_factor * (zr - prev_zr)

                    prev_xt = tv[0]
                    prev_yt = tv[1]
                    prev_zt = tv[2]
                    prev_zr = rv[2]

                    midX, midY, midZ, midR = 0, -0.0, marker_dist, 0
                    sMarX, sMarY, sMarZ, sMarR = 0.3, 0.15, 1, 30 * np.pi / 180
                    marX, marY, marZ, marR = 0.1, 0.05, 0.15, 15 * np.pi / 180

                    if simmode == "tello":
                        marX *= 0.8

                    _rl = 0
                    _fb = 0
                    _ud = 0
                    _rrl = 0

                    # Left-Right
                    if xt > midX + sMarX or xt < midX - sMarX:
                        _rl = np.sign(xt - midX)
                    elif xt > midX + marX or xt < midX - marX:
                        _rl = np.sign(xt - midX) * np.abs((xt - midX) / (sMarX - midX))

                    # if np.abs(xt - midX) < marX:   # fluctuating
                    #    marR = 25 * np.pi / 180

                    # Rotation
                    if zr > midR + sMarR or zr < midR - sMarR:
                        _rrl = np.sign(zr - midR)
                    elif zr > midR + marR or zr < midR - marR:
                        _rrl = np.sign(zr - midR) * np.abs((zr - midR) / (sMarR - midR))

                    if simmode == "dronesim":
                        _rrl /= 8
                    if simmode == "tello":
                        _rrl /= 4

                    if np.abs(xt - midX) > np.abs(0.25 * midZ) and np.sign(zt) == np.sign(
                            xt):  # shortcut for going left/right
                        _rrl = _rl / np.abs(xt * 3.75)
                        if zt > sMarZ + midZ:
                            _rrl /= np.abs(zt)
                    if np.abs(xt - midX) > np.abs(0.25 * midZ) and np.sign(zt) != np.sign(
                            xt):  # shortcut for going left/right
                        _rrl *= 3
                        _rl *= 2
                    #  if np.abs(xt - midX) < marX * 2 and np.abs(zr) > 30 * np.pi / 180:   # fluctuating
                    #    _rl = np.sign(zr)
                    #    _rrl *= 2

                    # Up-Down
                    if yt > midY + sMarY or yt < midY - sMarY:
                        _ud = -np.sign(yt - midY)
                    elif yt > midY + marY or yt < midY - marY:
                        _ud = -np.sign(yt - midY) * np.abs((yt - midY) / (sMarY - midY))

                    # Fore-Back
                    if zt > midZ + sMarZ or zt < midZ - sMarZ:
                        _fb = np.sign(zt - midZ)
                    elif zt > midZ + marZ or zt < midZ - marZ:
                        _fb = np.sign(zt - midZ) * np.abs((zt - midZ) / (sMarZ - midZ))

                    if training and _recording:
                        _rrl = round(joystick.get_axis(0), 5)
                        _rl = round(joystick.get_axis(2), 5)
                        _ud = round(joystick.get_axis(1), 5) * -1
                        _fb = round(joystick.get_axis(4), 5) * -1

                    if neural_control:
                        testing = 0
                        if testing == 1:
                            Yd = np.round(yt - midY, 2)  # tv[1] - midY
                            Zp = np.round(zt, 2)  # tv[2]
                            Zd = np.round(zt - midZ, 2)
                            Xp = np.round(xt, 2)  # tv[0]  # Input for X-pos
                            Rp = np.round(zr, 2)  # rv[2]  # Input for rotation
                            input_data = np.array([[Xp, Yd, Zp, Zp, Zd, Zd]])  # Replace with your actual input values

                            single_xp = np.array([Xp[0]])
                            single_yd = np.array([Yd[0]])
                            single_zp = np.array([Zp[0]])
                            single_zp2 = np.array([Zp[0]])
                            single_zd = np.array([Zd[0]])
                            single_rp = np.array([Rp[0]])

                            # Make predictions
                            predictions = full_model.predict( [Xp[0], Yd[0], Zp[0], Zp[0], Rp[0], Zd[0]])

                            # Separate the predictions for each output
                            #_rl, _ud, _fb, _rrl = predictions[0][0], predictions[0][1], predictions[0][2], \
                            #predictions[0][3]

                            # Make predictions
                            #predictions = model.predict(input_data)

                            # Extract the individual outputs
                            _rl = predictions[0]
                            _ud = predictions[1]
                            _fb = predictions[2]
                            #_rrl = predictions[3]
                        elif testing == 2:
                            Yd = np.round(yt - midY, 2)  # tv[1] - midY
                            Zp = np.round(zt, 2)  # tv[2]
                            Zd = np.round(zt - midZ, 2)
                            Xp = np.round(xt, 2)  # tv[0]  # Input for X-pos
                            Rp = np.round(zr, 2)  # rv[2]  # Input for rotation
                            inputxr = np.array([0.0, 0.0, 0.0])
                            inputy = np.array([0.0, 0.0])
                            inputz = np.array([0.0])
                            print(f"inputxr: {inputxr.shape}")
                            print(f"inputy: {inputy.shape}")
                            print(f"inputz: {inputz.shape}")
                            xrpredictions = combined_model.predict([[[inputxr], [inputy], [inputz]]])
                            #predictions = combined_model.predict([Xp, Rp, Zp], [Yd, Zp], Zd)
                            _rl = xrpredictions[0]
                            _ud = xrpredictions[1]
                            _fb = xrpredictions[2]
                            _rrl = xrpredictions[3]
                        else:
                            Yp = np.round(yt - midY, 2)  # tv[1] - midY
                            Zp = np.round(zt, 2)  # tv[2]
                            #input_y = np.array([[Yp, Zp]])
                            #predicted_outputs_y = model_y.predict(input_y)
                            #_ud = predicted_outputs_y[0]
                            neuralY_thread = threading.Thread(target=neural_y, args=(Yp, Zp))
                            neuralY_thread.start()

                            Zp = np.round(zt - midZ, 2)  # tv[2] - midZ
                            input_z = Zp
                            predicted_outputs_z = model_z.predict(input_z)
                            _fb = predicted_outputs_z
                            #neuralZ_thread = threading.Thread(target=neural_z, args=(Zp,))
                            #neuralZ_thread.start()

                            Xp = np.round(xt, 2)  # tv[0]  # Input for X-pos
                            Rp = np.round(zr, 2)  # rv[2]  # Input for rotation
                            Zp = np.round(zt, 2)  # tv[2]  # Input for Z-position
                            input_xr = np.array([[Xp, Rp, Zp]])
                            predicted_outputs_xr = model_xr.predict(input_xr)
                            _rl, _rrl = predicted_outputs_xr[0]
                            # neuralXR_thread = threading.Thread(target=neuralXR, args=(Xp, Rp, Zp))
                            # neuralXR_thread.start()

                    _mvec = _rl, _fb, _ud, _rrl
                    _move(_mvec)

                    if _rl != 0 or _rrl != 0 and trainDir == "xr":
                        _prerecord = False
                    if _ud != 0 and trainDir == "y":
                        _prerecord = False
                    if _fb != 0 and trainDir == "z":
                        _prerecord = False

                    # if _recording and (_rl != 0 or _rrl != 0):  # avoid harness late zeroes
                    if _recording and not _prerecord:  # can harness late zeroes

                        balance: bool = True
                        if tv[0] is None or tv[1] is None or tv[2] is None or rv[2] is None:
                            print("[Program] Error: Data not balanced. Review append for addend.")
                            balance = False

                        arXpt = np.append(arXpt, np.round(tv[0], 2)) if balance else arXpt
                        arYpt = np.append(arYpt, np.round(tv[1], 2)) if balance and trainDir != "y" else arYpt
                        arZpt = np.append(arZpt, np.round(tv[2], 2)) if balance and trainDir != "z" else arZpt
                        arRpt = np.append(arRpt, np.round(rv[2], 2)) if balance else arRpt
                        arXct = np.append(arXct, np.round(_rl, 2)) if balance else arXct
                        arYct = np.append(arYct, np.round(_ud, 2)) if balance else arYct
                        arZct = np.append(arZct, np.round(_fb, 2)) if balance else arZct
                        arRct = np.append(arRct, np.round(_rrl, 2)) if balance else arRct

                        if trainDir == "y":
                            arYpt = np.append(arYpt, np.round(tv[1] - midY, 2)) if balance else arYpt
                        if trainDir == "z":
                            arZpt = np.append(arZpt, np.round(tv[2] - midZ, 2)) if balance else arZpt

                # ---END OF AUTO-MOVEMENT-------------------------------------------------------------------------------

            else:  # Target ArUco Not Found: kill locked actions
                if _flying:
                    _mvec = 0, 0, 0, 0
                    _move(_mvec)
                    # if simmode == "tello" and _flying and _automove:
                    #    tello.send_rc_control(0, 0, 0, 0)
                    #    #_move(_mvec)
                    # if (simmode == "none" or simmode == "dronesim") and _flying and _automove:
                    #    _move(_mvec)
                cv.imshow("ArUco Marker Detection", frame)

        else:
            # No Markers Found
            if _automove:
                _mvec = (0, 0, 0, 0)
                _move(_mvec)
            cv.imshow("ArUco Marker Detection", frame)

        # Land and takeoff
        _takeofflandcount += 1  # avoid sudden switch of land - takeoff
        if keyboard.is_pressed(keytakeoffland) or (joystick.get_button(6) and joystick.get_button(7)):
            if _zkeyisup:
                if _takeofflandcount * frame_time < _takeofflandwait:
                    if keyboard.is_pressed(keytakeoffland) or (joystick.get_button(6) and joystick.get_button(7)):
                        if _flying:
                            play_this("please wait before landing")
                            print(f"[Program] Please wait for "
                                  f"{round(_takeofflandwait - _takeofflandcount * frame_time, 3)} "
                                  f"seconds before commanding to land.")
                        else:
                            play_this("please wait before flying")
                            print(f"[Program] Please wait for "
                                  f"{round(_takeofflandwait - _takeofflandcount * frame_time, 3)} "
                                  f"seconds before commanding to takeoff.")
                    _zkeyisup = False
                else:
                    _takeofflandcount = 0
                    play_this("switched to manual", 1.3) if _automove else None
                    print("[Program]: Control is manual.") if _automove else None
                    _automove = False
                    if _flying:  # for landing
                        play_this("tello is landing")
                        print("[Program] Tello is landing.")
                        _flying = False
                        _mvec = (0, 0, 0, 0)
                        _move(_mvec)
                        if simmode == "tello" and _realtelloflight:
                            tello.land()
                        print("[Program] Tello has landed.") if simmode == "tello" else None
                    else:
                        print("[Program] Tello is taking off.") if simmode == "tello" else None
                        if _realtelloflight and simmode == "tello":
                            tello.takeoff()
                        _flying = True
                        play_this("tello is flying")
                        print("[Program] Tello is flying")
                    _zkeyisup = False
        else:
            _zkeyisup = True

        # Display the frame
        #cv.imshow("ArUco Marker Detection", frame) if _idle else None

    else:  # No frame
        # freeze after a running video
        if initiated_frame and not frozen_frame:
            play_this("video has frozen")
            print("[Program] Video frame has frozen.")
            frozen_frame = True
            freeze_start_time = time.time()
        # freeze before running video
        if (not initiated_frame or frozen_frame) and time.time() - freeze_start_time >= freeze_time_limit:
            play_this("failed video get")
            print("[Program] Failed to get the video.")
            frozen_frame = True
            time.sleep(1.8)
            break

    frame_time = time.time() - testtimestart
    no_sleep_frame_time = frame_time - sleep_time
    #print(f"time : {frame_time}") if neural_control else None
# ---------------------------END MAIN RUN-------------------------------------------------------------------------------

play_this("closing the program")
print("[Program] Closing the program...")

# Release the camera and close all OpenCV windows
_mvec = 0, 0, 0, 0
_stop("x", _mvec)
_stop("y", _mvec)
_stop("z", _mvec)
_stop("r", _mvec)
endSpeechTime = 1.5
if simmode == "tello":
    tello.land() if _flying and _realtelloflight else None
    tello.streamoff()
    tello.end()
if simmode in ["dronesim", "tello", "none"]:
    if _flying:
        keyboard.press(keytakeoffland)
        time.sleep(0.01)
        keyboard.release(keytakeoffland)
        play_this("tello is landing", 1.5)
        endSpeechTime += 1.5
        print("[Program] Automatic landing initiated.")
    vj1.reset() if simmode == "dronesim" else None
    # py_window = gw.getWindowsWithTitle("ArUcoFollower.py")
    # py_window[0].activate()
if _control == "joystick":
    pygame.quit()
cap.release() if vidsource in ["none", "obs", "defcam", "webcam"] else None
sct.close() if vidsource in ["dronesim", "dronesim_phone"] else None

play_this("successfully closed", endSpeechTime)
print("[Program] Program successfully closed.")
cv.destroyAllWindows()
