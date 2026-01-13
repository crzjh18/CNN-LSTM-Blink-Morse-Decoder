import cv2
import os
import shutil

# ================= CONFIGURATION =================
# Point this to your CLOSED folder
TARGET_FOLDER = r"S:\VSCode Projects\MediaFace\dataset_dynamic_aligned\closed"

# Where to move the "intruders" (images that actually have open eyes)
TRASH_FOLDER = r"S:\VSCode Projects\Backup Code\Training dataset blurred images\blurry images\closed"
# =================================================

def clean_open_eyes_from_closed_folder():
    # 1. Load the Detector
    cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    
    if eye_cascade.empty():
        print("Error: Could not find haar cascade xml file.")
        return

    print(f"Scanning: Removing OPEN eyes from {TARGET_FOLDER}")
    os.makedirs(TRASH_FOLDER, exist_ok=True)
    
    files = [f for f in os.listdir(TARGET_FOLDER) if f.lower().endswith('.jpg')]
    moved_count = 0
    kept_count = 0

    for file in files:
        file_path = os.path.join(TARGET_FOLDER, file)
        
        img = cv2.imread(file_path)
        if img is None: continue
        
        # 2. Pre-processing
        # Resize to give the detector a better chance at seeing the eye details
        large_img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)

        # 3. Detect Eyes
        # minNeighbors=2: Balanced. 
        # Set to 1 if you want to be VERY strict and remove anything that even looks like an eye.
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=20, minSize=(30, 30))

        # 4. Decision Logic
        if len(eyes) > 0:
            # DETECTION = OPEN EYE.
            # Since this is a "Closed" folder, this is an error. Move it.
            print(f"MOVING: {file} (Detected open eye)")
            try:
                shutil.move(file_path, os.path.join(TRASH_FOLDER, file))
                moved_count += 1
            except Exception as e:
                print(f"Error moving: {e}")
        else:
            # NO DETECTION = CLOSED EYE (or garbage). Keep it.
            kept_count += 1

    print(f"\nScan Complete.")
    print(f"Kept:  {kept_count} (Likely Closed)")
    print(f"Moved: {moved_count} (Detected Open)")
    print(f"Check '{TRASH_FOLDER}' to verify.")

if __name__ == "__main__":
    clean_open_eyes_from_closed_folder()