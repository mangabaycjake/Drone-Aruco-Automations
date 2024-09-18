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
import socket

play_this("finished importing")
print("[Program] Finished importing libraries.")
transition_time_start: float = time.time()

# CHANGEABLE PARAMETERS-------------------------------------------------------------------------------------------------
vidsource = "dronesim"  # defcam, webcam, obs, tello,dronesim, dronesim_phone
simmode = "dronesim"  # none, tello, dronesim
MARKER_SIZE: float = 0.5  # meters (measure your printed marker size)
target_height: float = 2  # meters
training: bool = False
neural_control: bool = True
neural_control = False if training else True

path_desc = "zigzag2"
trainDir = "xr"
comp = "laptop"  # pc, laptop
# ----------------------------------------------------------------------------------------------------------------------
maxD = 10
_realtelloflight: bool = True
marker_dist = 1.5  # meters
_automove = False  # only if simmode is tello
key_stop = "esc"
key_ctrlswitch = 'c'
keytakeoffland = 'z'
jsHighS: int = 5  # high speed at R1 of joystick
_fps: float = 10  # fps
_takeofflandwait: int = 3  # in seconds, wait before taking-off / landing
marker_count: int = 9

# Parameters prompt
print(f"[Program] Video Source: {vidsource} | Simulation Mode: {simmode} | Real Tello Flight: {_realtelloflight}")
print(f"[Program] Press [{key_stop}, {key_ctrlswitch}, {keytakeoffland}] to [Exit, Toggle AutoFlight, Takeoff/Land")

# INITIALIZATIONS
_mainsleep: float = 1 / _fps
cap = None
tello = Tello() if simmode == "tello" else None
pygame.init()
pygame.joystick.init()
frame = None
vj = None
freeze_time_limit = 5
jsNeu: int = 16384  # Neutral position (centered)
jsMax: int = 32768  # Maximum position
sct = mss.mss() if vidsource in ["dronesim", "dronesim_phone"] else None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set to '2' to suppress INFO logs in tensorflow
_flying = False
_rl, _fb, _ud, _rrl = 0.0, 0.0, 0.0, 0.0
vj1 = pyvjoy.VJoyDevice(1)
frame_time = _mainsleep
running = True


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
parameters.minMarkerPerimeterRate = 0.01 / (float(MARKER_SIZE / 0.05) * 10)  # Adjust to a smaller value
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
            vj1.set_axis(pyvjoy.HID_USAGE_Z, jsNeu)
    if axis == "y":
        if simmode == "tello":
            ud = 0
        if simmode == "dronesim":
            vj1.set_axis(pyvjoy.HID_USAGE_Y, jsNeu)
    if axis == "z":
        if simmode == "tello":
            fb = 0
        if simmode == "dronesim":
            vj1.set_axis(pyvjoy.HID_USAGE_RY, jsNeu)
    if axis == "r":
        if simmode == "tello":
            rrl = 0
        if simmode == "dronesim":
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


def max1_scale(_val, _axis):
    global maxD
    if _axis == "r":
        _val /= (np.pi / 2)
    else:
        if np.abs(_val) > maxD:
            _val = np.sign(_val) * maxD
        _val /= maxD
    return _val

training_data_t = np.empty((0, (marker_count * 2) + 4), dtype=float)
training_data = np.empty((0, (marker_count * 2) + 4), dtype=float)


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
        subprocess.Popen("Drone Simulation - Shortcut.exe.lnk", shell=True)
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


# Unity Coms
unityData = None
def receive_data(_client_socket):
    while True:
        _data = _client_socket.recv(1024).decode()
        if not _data or not running:
            break  # Exit the loop when the connection is closed
        global unityData
        unityData = _data


def ask_unity(what):
    what = "" + what
    client_socket.send(what.encode())
    global unityData
    return unityData


if simmode in ["dronesim", "dronesim_phone"]:
    server_ip = socket.gethostbyname(socket.gethostname())  # Replace with the Unity server's IP address
    server_port = 12345
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    print("Connected to Unity server")
    receive_thread = threading.Thread(target=receive_data, args=(client_socket,))
    receive_thread.start()
    # Determine if already flying
    unityData = client_socket.recv(1024).decode()
    print(f"unity: unityData")
    print("ffffffffff")
    if round(float(unityData), 3) > 0.1:
        _flying = True
        print("[Program] Tello is flying.")
        play_this("tello is flying")
        time.sleep(1)

if neural_control:
    traveller_model_path = script_directory + "/calibration_data/" + vidsource + "/path_" + path_desc + "/traveller_model_" + vidsource + '_' + path_desc + ".h5"
    traveller_model = keras.models.load_model(traveller_model_path)
    traveller_model.summary()

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



# MAIN LOOP ____________________________________________________________________________________________________________
while True:

    testtimestart = time.time()
    sleep_time = 0
    deltaTime = time.time() - prevTime
    prevTime = time.time()

    # Exit the loop
    if cv.waitKey(1) & keyboard.is_pressed(key_stop):  # if the stop key is pressed
        break

    if joystick.get_button(4) and joystick.get_button(6):  # L1 and L2
        break

    for event in pygame.event.get():  # Needed it for joystick
        if event.type == pygame.QUIT:
            _ = None

    # Send data to unity
    sendData = ""
    if sendData != "":
        client_socket.send(sendData.encode())

    # Switch Transfer control to manual / automatic
    if keyboard.is_pressed(key_ctrlswitch) or (joystick.get_button(4) and not joystick.get_button(6)):  # avoid @ close
        if _ckeyisup:
            if _flying:
                if _automove:
                    _automove = False
                    _mvec = 0, 0, 0, 0
                    _move(_mvec)
                    play_this("switched to manual") if not training else None
                    print("[Program] Control is manual.") if not training else None
                    if training:
                        _recording = False
                        play_this("paused")
                        print("[Program] Paused recording.")
                else:
                    _automove = True
                    play_this("automatic control") if not training else None
                    print("[Program] Control is automatic.") if not training else None
                    if training:  # Auto append
                        if training_data_t.shape[0] != 0:
                            training_data = np.vstack((training_data, training_data_t))
                            print(f"[Program] Added data: {training_data_t.shape[0]} | Total: {training_data.shape[0]}")
                            training_data_t = np.empty((0, (marker_count * 2) + 4), dtype=float)
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
        # Append last record
        if joystick.get_button(0):
            if _appendkeyisup:
                _appendkeyisup = False
                if training_data_t.shape[0] != 0:
                    if not _recording:
                        training_data = np.vstack((training_data, training_data_t))
                        print(f"[Program] Added data: {training_data_t.shape[0]} | Total: {training_data.shape[0]}")
                        training_data_t = np.empty((0, (marker_count * 2) + 4), dtype=float)
                        play_this("appended")
                        print("[Program] Last recording was appended.")
                    else:
                        play_this("pause first before appending")
                        print("[Program] Pause the recording first.")
                else:
                    print("[Program] Nothing to append.")
                    play_this("nothing to append")
        else:
            _appendkeyisup = True

        # Delete pending record
        if joystick.get_button(1):
            if _delkeyisup:
                _delkeyisup = False
                if not _recording:
                    if training_data_t.shape[0] != 0:
                        training_data_t = np.empty((0, (marker_count * 2) + 4), dtype=float)
                        play_this("deleted")
                        print("[Program] Last recording was deleted.")
                    else:
                        print("[Program] Nothing to delete.")
                        play_this("nothing to delete")
                else:
                    play_this("pause first before deleting")
                    print("[Program] Pause the recording first.")
        else:
            _delkeyisup = True

        # Save as NPZ
        if joystick.get_button(2):
            if _savekeyisup:
                _savekeyisup = False
                save_data_path = script_directory + "/calibration_data/" + vidsource + "/path_" + path_desc + "/traveller_training_" + vidsource + '_' + path_desc

                # Define column names
                #column_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
                # Initialize an empty list to store the column names
                column_names = []

                # Add 'R' at the beginning of the list
                column_names.append('R')

                # Create column names based on the maker_count
                for i in range(marker_count):
                    column_names.extend([f"X{i}", f"Z{i}"])

                    # Add additional columns
                column_names.extend(["Xc", "Zc", "Rc"])

                # Create a structured array with named columns
                #structured_array = np.core.records.fromarrays(training_data, names=column_names)

                # Save the structured array to an NPZ file
                np.savez(save_data_path + ".npz", data=training_data)
                np.savetxt(save_data_path + '.csv', training_data, delimiter=',', header=','.join(column_names),
                           comments='')

                play_this("NPZ saved")
                print("[Program] Saved as npz.")
        else:
            _savekeyisup = True

    # ---Manual Movement------------------------------------------------------------------------------------------------
    if (not _automove) and (simmode == "tello" or simmode == "none"):
        if _flying:
            _rl: float = 0
            _fb: float = 0
            _ud: float = 0
            _rrl: float = 0
            # keyboard controls
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

            # Joystick controls
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

        sleep_time = _mainsleep - no_sleep_frame_time
        if sleep_time < 0:
            sleep_time = 0
        #time.sleep(sleep_time)

        # IF MARKERS ARE DETECTED
        marker_present = False
        if ids is not None:
            for _id in ids:
                if _id < marker_count:
                    marker_present = True
        #if ids is not None:
        if marker_present:
            cv.aruco.drawDetectedMarkers(frame, corners)
            # New Pose Estimate
            tv = np.zeros((3, marker_count))
            rv = np.zeros((3, marker_count))
            rv_ = None
            for i in range(0, len(ids)):
                if ids[i] < marker_count:
                    _, _rv, _tv = cv.solvePnP(objPoints, corners[i], cam_mat, dist_coef, False,
                                        cv.SOLVEPNP_IPPE_SQUARE)
                    cv.drawFrameAxes(frame, cam_mat, dist_coef, _rv, _tv, 0.1)
                    rv[:, ids[i]] = _rv
                    tv[:, ids[i]] = _tv
                    if rv[1][ids[i]] != 0:
                        rv_ = rv[1][ids[i]]
            #print(rv[1][ids[0]])
            cv.imshow("ArUco Marker Detection", frame)

            # ---AUTO MOVEMENT ---

            xpos = np.zeros(marker_count)
            zpos = np.zeros(marker_count)


            inputs = np.array(np.round([max1_scale(rv_[0], "r")], 3))  # First element is the rotation vector along the vertical axis

            for _id in ids:
                if _id < marker_count:
                    xpos[_id] = np.round(max1_scale(tv[0][_id], "x"), 3)
                    zpos[_id] = np.round(max1_scale(tv[2][_id], "y"), 3)

            for i in range(marker_count):
                inputs = np.hstack((inputs, xpos[i]))
                inputs = np.hstack((inputs, zpos[i]))

            if _automove and _flying:

                if training and _recording:
                    _rrl = round(joystick.get_axis(0), 5)
                    _rl = round(joystick.get_axis(2), 5)
                    _ud = round(joystick.get_axis(1), 5) * -1
                    _fb = round(joystick.get_axis(4), 5) * -1
                    outputs = np.array([_rl, _fb, _rrl])

                    if _rl != 0 or _fb != 0 or _rrl != 0:  # Moving by control
                        _prerecord = False

                    if not _prerecord:
                        if inputs.shape[0] == (marker_count * 2) + 1:
                            input_output = np.hstack((inputs, outputs))
                            training_data_t = np.vstack((training_data_t, input_output))

                if neural_control:
                    inputs = np.atleast_2d(inputs)
                    predictions = traveller_model.predict(inputs)
                    _rl, _fb, _rrl = predictions[0]

                    #input_xr = np.array([[Xp, Rp, Zp]])
                    #predicted_outputs_xr = model_xr.predict(input_xr)
                    #_rl, _rrl = predicted_outputs_xr[0]
                    # neuralXR_thread = threading.Thread(target=neuralXR, args=(Xp, Rp, Zp))
                    # neuralXR_thread.start()
                altitude = round(float(unityData), 2)  # stabilize at at fixed altitude
                if _automove and _flying and altitude != target_height:
                    _ud = 0.5 * np.sign(target_height - altitude)

                _mvec = _rl, _fb, _ud, _rrl
                _move(_mvec)

                # if _recording and (_rl != 0 or _rrl != 0):  # avoid harness late zeroes
                if _recording and not _prerecord:  # can harness late zeroes
                    None

            # ---END OF AUTO-MOVEMENT-------------------------------------------------------------------------------


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
                    if simmode in ["dronesim", "dronesim_phone"]:
                        _flying = True if float(unityData) > 0.1 else _flying
                    play_this("switched to manual", 1.3) if _automove else None
                    print("[Program] Control is manual.") if _automove else None
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
                        print("[Program] Tello is flying.")
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
running = False
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

pygame.quit()
cap.release() if vidsource in ["none", "obs", "defcam", "webcam"] else None
sct.close() if vidsource in ["dronesim", "dronesim_phone"] else None

# Close the sockets when done
client_socket.close() if simmode == ["dronesim", "dronesim_phone"] else None

play_this("successfully closed", endSpeechTime)
print("[Program] Program successfully closed.")
cv.destroyAllWindows()
