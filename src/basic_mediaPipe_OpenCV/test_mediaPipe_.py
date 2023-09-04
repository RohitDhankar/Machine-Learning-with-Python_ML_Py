# conda activate env2_det2
import cv2
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import subprocess
from datetime import datetime 

dt_time_now = datetime.now()
dt_time_now_US_Format = dt_time_now.strftime("_%m_%d_%Y_%H:%M:%S") 
#print(dt_time_now_US_Format)

save_temp_path = "./data_dir/output_dir"

def track_hands():
    """
    """
    cap = cv2.VideoCapture(0)
    print("--type---",cap)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    for iter_n in range(49):
    # while True:
        print("---ITER----",iter_n)
        success, init_img = cap.read()
        imgRGB = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
        print("--type----",type(imgRGB))

        # cap.release()
        # cv2.destroyAllWindows()

        #cv2.imshow("HAND_INIT", imgRGB)
        #cv2.imwrite(save_temp_path+"/"+str(dt_time_now_US_Format)+"_test_.png",imgRGB)

        results = hands.process(imgRGB)
        print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    print(id, lm)
                    h, w, c = init_img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id, cx, cy)
                    # if id == 12:
                    cv2.circle(init_img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                mpDraw.draw_landmarks(init_img, handLms, mpHands.HAND_CONNECTIONS)
        cv2.imwrite(save_temp_path+"/"+str(dt_time_now_US_Format)+"__"+str(iter_n)+"_rohit_.png",init_img)
        
        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime

        # #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        # cv2.imshow("Image", img)
        # cv2.waitKey(1)


def track_pose():
    """
    """
    mpPose = mp.solutions.pose
    pose = mpPose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    print("--type---",cap)

    num_landmarks =0
    loopCount = 0
    for iter_n in range(30):
    #while True:
        print("---ITER--POSE--",iter_n)
        #now doing this in a loop so we replay video indefintely
        # Initialize our video source. It can be a file or a webcam.
        # cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture('videos/BakingBrains_a.mp4')
        #cap = cv2.VideoCapture('videos/JustSomeMotion.mov')
        #cap = cv2.VideoCapture('videos/body made of water.mov')
        #cap = cv2.VideoCapture('videos/Coreografia.mov')
        #cap = cv2.VideoCapture('videos/Fred Astaire Oscars.mov')

        frameCount = 0
        loopCount = loopCount + 1

        for iter_k in range(30):
        #while True:
            #startTime = time.time()
            try:
                #
                success, init_img = cap.read()

                if isinstance(init_img, (np.ndarray, np.generic)):
                    frameCount = frameCount + 1
                    imgRGB = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
                    image_height, image_width, _ = imgRGB.shape
                    results = pose.process(imgRGB)
                    num_landmarks = len(results.pose_landmarks.landmark)
                    #print("height, width, num marks", image_height, image_width, num_landmarks)

                    if results.pose_landmarks:
                        print("---ITER--POSE---iter_k--",iter_k)
                        # draw landmark connection lines (skeleton)
                        mpDraw.draw_landmarks(init_img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                    cv2.imwrite(save_temp_path+"/"+str(dt_time_now_US_Format)+"_pose_"+str(iter_k)+str(iter_n)+"_rohit_.png",init_img)
                else:
                    print("--NO IMAGE ----ERROR--->")

            except Exception as err_get_init_img:
                print('-[ERROR]--err_get_init_img--\n',err_get_init_img)
                pass
                # client.send_message(f"/image-height", image_height)
                # client.send_message(f"/image-width", image_width)
                # client.send_message(f"/numLandmarks", num_landmarks)
                #print("height, width, num marks", image_height, image_width, num_landmarks)

                # for id, lm in enumerate(results.pose_landmarks.landmark):
                #     # Draw circles on the pose areas. This is purely for debugging
                #     #cx, cy = int(lm.x * image_width), int(lm.y * image_height)
                #     #cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

                #     point_name = pose_id_to_name.get(id)
                #     # Send our values over OSC once w/all 3 values in one msg
                #     # this saves in the comm layers at expense of parsing in TD
                #     # note using uv screen space soords rather than xyz
                #     # and z is actually Confidence
                #     #client.send_message(f"/p1/{point_name}", [lm.x,lm.y,lm.z])
                #     if point_name == 'handtip_l':
                #         print("handtip_l: ",lm.x,lm.y,lm.z)
                #     # could send as 3 OSC msgs to better match kinect names
                #     #client.send_message(f"/p1/{point_name}:u", lm.x)
                #     #client.send_message(f"/p1/{point_name}:v", lm.y)#adjustY(lm.y, image_width, image_height))
                #     #client.send_message(f"/p1/{point_name}:tz", lm.z)

        #     endTime = time.time()
        #     elapsedTime = endTime - startTime
        #     fps = 1.0/elapsedTime
        #     print("Loop %d Frame %d Rate %.2f Elapsed %.2f" %
        #         (loopCount,frameCount,fps,elapsedTime))
        #     cv2.imshow("Image", img)
        #     cv2.waitKey(1)
        # print("Loop Video")



# # names of kinect and mediapipe landmarks
# landmarkNames = [
#     'head',
#     'mp_eye_inner_l',
#     'eye_l',
#     'mp_eye_outer_l',
#     'mp_eye_inner_r',
#     'eye_r',
#     'mp_eye_outer_e',
#     'mp_ear_l',
#     'mp_ear_r',
#     'mp_mouth_l',
#     'mp_mouth_r',
#     'shoulder_l',
#     'shoulder_r',
#     'elbow_l',
#     'elbow_r',
#     'wrist_l',
#     'wrist_r',
#     'mp_pinky_l',
#     'mp_pinky_r',
#     'handtip_l',
#     'handtip_r',
#     'thumb_l',
#     'thumb_r',
#     'hip_l',
#     'hip_r',
#     'knee_l',
#     'knee_r',
#     'ankle_l',
#     'ankle_r',
#     'mp_heel_l',
#     'mp_heel_r',
#     'foot_l',
#     'foot_r',
# #   'shoulder_c',
# #   'spine'
# ]

# # Create a map to look up name given id
# pose_id_to_name = {i: name for i, name in enumerate(landmarkNames)}

# def test_name_map(id=6):
#     name = pose_id_to_name.get(id)
#     if name:
#         print(f"The name for ID {id} is '{name}'.")
#     else:
#         print(f"No name found for ID {id}.")
# test_name_map()
    
if __name__ == "__main__":
    track_hands()
    track_pose()


## https://hackaday.io/project/188345-pose2art-smartcam-to-touchdesigner-unity-via-osc
## https://github.com/MauiJerry/Pose2Art
