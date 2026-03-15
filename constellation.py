import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

WIDTH = 1280
HEIGHT = 720
MAX_PARTICLES = 500


# -------------------------------
# PARTICLE INITIALIZATION
# -------------------------------

def init_particles(n):
    pos = np.random.rand(n,2) * [WIDTH,HEIGHT]
    vel = np.zeros((n,2))
    active = np.zeros(n,dtype=bool)

    colors = np.zeros((n,3))
    colors[:] = [0,255,255]

    return pos,vel,active,colors


# -------------------------------
# MEDIAPIPE SETUP
# -------------------------------

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose = PoseLandmarker.create_from_options(options)


# -------------------------------
# VIDEO PROCESSOR
# -------------------------------

class PoseProcessor(VideoProcessorBase):

    def __init__(self):

        self.pos,self.vel,self.active,self.colors = init_particles(MAX_PARTICLES)

        self.person_detected=False
        self.scan_y=0
        self.scanning=False

        self.wrist_history=deque(maxlen=10)

        self.G_BASE=5000


    def recv(self,frame):

        img = frame.to_ndarray(format="bgr24")

        img=cv2.resize(img,(WIDTH,HEIGHT))
        img=cv2.flip(img,1)

        frame_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        timestamp_ms=int(time.time()*1000)

        mp_image=mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        results=pose.detect_for_video(mp_image,timestamp_ms)

        targets=[]
        black_hole=False
        supernova=False

        target_color=[0,255,255]

        bh_center=None


        # ---------------------------------
        # POSE DETECTION
        # ---------------------------------

        if results.pose_landmarks:

            if not self.person_detected:
                self.person_detected=True
                self.scanning=True

            landmarks=results.pose_landmarks[0]

            for lm in landmarks:

                if lm.visibility>0.3:

                    x=int(lm.x*WIDTH)
                    y=int(lm.y*HEIGHT)

                    targets.append([x,y])

                    cv2.circle(img,(x,y),3,(0,0,255),-1)

            LEFT_WRIST=15
            RIGHT_WRIST=16

            lw=landmarks[LEFT_WRIST]
            rw=landmarks[RIGHT_WRIST]

            if lw.visibility>0.5 and rw.visibility>0.5:

                lw_pos=np.array([lw.x*WIDTH,lw.y*HEIGHT])
                rw_pos=np.array([rw.x*WIDTH,rw.y*HEIGHT])

                self.wrist_history.append((lw_pos,rw_pos))

                wrist_dist=np.linalg.norm(lw_pos-rw_pos)/WIDTH

                if wrist_dist<0.1:
                    black_hole=True
                    bh_center=(lw_pos+rw_pos)/2


        targets=np.array(targets) if len(targets)>0 else np.array([])


        # ---------------------------------
        # SCANNING BAR
        # ---------------------------------

        if self.scanning:

            self.scan_y+=4

            if self.scan_y>HEIGHT:
                self.scanning=False

            mask=self.pos[:,1]<self.scan_y
            self.active[mask]=True

            cv2.line(
                img,
                (0,int(self.scan_y)),
                (WIDTH,int(self.scan_y)),
                (255,255,0),
                3
            )


        # ---------------------------------
        # PARTICLE PHYSICS
        # ---------------------------------

        active_mask=self.active

        if len(targets)>0 and np.any(active_mask):

            active_pos=self.pos[active_mask]
            active_vel=self.vel[active_mask]

            diffs=targets[np.newaxis,:,:]-active_pos[:,np.newaxis,:]

            dists_sq=np.sum(diffs**2,axis=2)

            closest_idx=np.argmin(dists_sq,axis=1)

            closest_diff=diffs[np.arange(len(active_pos)),closest_idx]

            closest_dist_sq=dists_sq[np.arange(len(active_pos)),closest_idx]+20

            closest_dist=np.sqrt(closest_dist_sq)

            error=closest_dist-40

            force=np.clip(0.15*error,-10,10)

            acc=(closest_diff/closest_dist[:,None])*force[:,None]

            active_vel+=acc

            active_vel*=0.95

            active_pos+=active_vel

            self.pos[active_mask]=active_pos
            self.vel[active_mask]=active_vel


        # ---------------------------------
        # DRAW PARTICLES
        # ---------------------------------

        active_indices=np.where(self.active)[0]

        for i in active_indices:

            x=int(self.pos[i,0])
            y=int(self.pos[i,1])

            if 0<=x<WIDTH and 0<=y<HEIGHT:

                c=(
                    int(self.colors[i,0]),
                    int(self.colors[i,1]),
                    int(self.colors[i,2])
                )

                cv2.circle(img,(x,y),2,c,-1)


        # ---------------------------------
        # TEXT
        # ---------------------------------

        if self.person_detected:

            cv2.putText(
                img,
                "BIOMETRIC SIGNATURE DETECTED",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )


        return av.VideoFrame.from_ndarray(img,format="bgr24")



# -------------------------------
# STREAMLIT APP
# -------------------------------

st.title("N-Body Constellation")

st.write("Move your body to interact with particles.")

webrtc_streamer(
    key="pose",
    video_processor_factory=PoseProcessor,
    media_stream_constraints={
        "video":True,
        "audio":False
    },
)
