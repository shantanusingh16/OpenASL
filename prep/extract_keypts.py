import os
import cv2
import argparse
import numpy as np
import glob

from p_tqdm import p_map
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

OUTPUT_KEYPTS_DIR = '/scratch/shantanu/openasl_cropped_keypts'


def extract_keypts(frame_dir):
    image_fps = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
    filename = os.path.basename(frame_dir)

    body_fp = os.path.join(OUTPUT_KEYPTS_DIR, 'body', f'{filename}.npy')
    hand_fp = os.path.join(OUTPUT_KEYPTS_DIR, 'hand', f'{filename}.npy')
    face_fp = os.path.join(OUTPUT_KEYPTS_DIR, 'face', f'{filename}.npy')

    ## Check if extraction is already done
    body_exist = os.path.exists(body_fp)
    hand_exist = os.path.exists(hand_fp)
    face_exist = os.path.exists(face_fp)
    if body_exist and hand_exist and face_exist:
        print(f'Skipping {filename}, data already exists')
        return
    else:
        if body_exist:
            os.remove(body_fp)
        if hand_exist:
            os.remove(hand_exist)
        if face_exist:
            os.remove(face_exist)

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=True, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    body_data = np.zeros((len(image_fps), 33, 5), dtype=np.float32) 
    face_data = np.zeros((len(image_fps), 478, 5), dtype=np.float32) 
    hand_data = np.zeros((len(image_fps), 2, 21, 5), dtype=np.float32) 
    for iidx, image_fp in enumerate(image_fps):
        image = cv2.imread(image_fp, -1)
        if image is None:
            print(f'File not found: {image_fp}')
            continue

        body_kpts = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if body_kpts.pose_landmarks:
            for lidx, lm in enumerate(body_kpts.pose_landmarks.landmark):
                body_data[iidx, lidx] = np.array([lm.x, lm.y, lm.z, lm.presence, lm.visibility], dtype=np.float32)

        face_kpts = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if face_kpts.multi_face_landmarks:
            for lidx, lm in enumerate(face_kpts.multi_face_landmarks[0].landmark):
                face_data[iidx, lidx] = np.array([lm.x, lm.y, lm.z, lm.presence, lm.visibility], dtype=np.float32)

        hand_kpts = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if hand_kpts.multi_hand_landmarks:
            for hidx, hand in enumerate(hand_kpts.multi_hand_landmarks):
                if hidx > 1:
                    break
                sidx = (hand_kpts.multi_handedness[hidx].classification[0].label == 'Right') * 1 #side-index, left-0, right-1
                for lidx, lm in enumerate(hand.landmark):
                    hand_data[iidx, sidx, lidx] = np.array([lm.x, lm.y, lm.z, lm.presence, lm.visibility], dtype=np.float32)

    pose.close()
    face_mesh.close()
    hands.close()
    np.save(body_fp, body_data)
    np.save(face_fp, face_data)
    np.save(hand_fp, hand_data)




def main():
    parser = argparse.ArgumentParser(description='extract keypoints using mediapipe', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw', type=str, help='raw video dir')
    args = parser.parse_args()

    parent_dir = args.raw
    if not (os.path.exists(parent_dir) or os.path.isdir(parent_dir)):
        raise Exception(f"Provided incorrect raw directory: {parent_dir}")

    frame_dirs = [dirpath for dirpath in glob.glob(os.path.join(parent_dir, '*')) if os.path.isdir(dirpath)]
    print(len(frame_dirs))

    # frame_dirs = frame_dirs[:10]

    p_map(extract_keypts, frame_dirs, num_cpus=10)


if __name__ == '__main__':
    os.makedirs(OUTPUT_KEYPTS_DIR, exist_ok=True)
    for dirname in ['body', 'face', 'hand']:
        os.makedirs(os.path.join(OUTPUT_KEYPTS_DIR, dirname), exist_ok=True)
    main()
