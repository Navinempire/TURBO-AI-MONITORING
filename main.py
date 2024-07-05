import streamlit as st
import os
import shutil
import cv2
from Yolo_predictions1 import YOLO_pred
from PIL import Image
import time
import urllib.request
import numpy as np
yolo = YOLO_pred("D:\AI BOOSTERS\opencv\data.yaml", "D:\\AI BOOSTERS\\opencv\\Model17\\weights\\best.onnx")

st.title("Turbo AI Monitoring")

style = """
<style>
    .appview-container {
        background-image: url("D:\\AI BOOSTERS\\opencv\\static\\img\\body.png");
    }
</style>
"""

st.write(style,unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file")

if st.button("Process video"):
    if uploaded_file is not None:
        with open("uploaded_file.mp4","wb") as out_file:
            shutil.copyfileobj(uploaded_file,out_file)
        
        cap = cv2.VideoCapture("./uploaded_file.mp4")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("detectedvideo.mp4", fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if ret == False:
                print("Unable to read the video.")
                break
                
            pred_image = yolo.predictions(frame)
            
            # Write the frame with predictions to the output video
            out.write(pred_image)
            
            cv2.imshow('yolo', pred_image)
            if cv2.waitKey(1) == 27:
                break

        # Release video capture and writer
        cap.release()
        out.release()
        cv2.destroyAllWindows()



        with open("detectedvideo.mp4", "rb") as f:
            video_bytes = f.read()
        st.download_button(
            label="Download processed video",
            data=video_bytes,
            mime="video/mp4"
            )

   

    else:
        st.write("NO file uploaded")
        

elif st.button("Live Process"):
    url = "http://192.168.137.81/cam-hi.jpg"

    timestamp = int(time.time())
    output_dir = f'video_store/video_capture{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0  # Initialize frame counter

    while True:
        try:
            img_resp = urllib.request.urlopen(url)
            img_array = np.asarray(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Unable to fetch frame: {e}")
            continue

        # Save each frame as an image inside the unique folder
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        pred_image = yolo.predictions(frame)

        cv2.imshow('yolo', pred_image)
        if cv2.waitKey(1) == 27:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_count += 1

    cv2.destroyAllWindows()



