
import argparse
import os
import cv2
import pandas as pd
from roboflow import Roboflow
import supervision as sv
import sys

parser = argparse.ArgumentParser(description = "Runs inference integrated with roboflow")
parser.add_argument("--output_folder", type=str, default = "Output", help="Name of folder to store Inferences")
parser.add_argument("--image_dir", type=str, default="Images", help="Path to image directory of images you want to run inference on")
parser.add_argument("--api", type=str, default="xxxxxx", help="API Key for Roboflow")
parser.add_argument("--proj_name", type=str, default="kaziga-open-flowers", help="Project Folder Name")
parser.add_argument("--model_num", type=int, default=6, help="Model Number")
arg = parser.parse_args()

# Config
API_KEY = arg.api
PROJECT_NAME = arg.proj_name
MODEL_VERSION = arg.model_num
IMAGE_FOLDER = arg.image_dir
OUTPUT_FOLDER = arg.output_folder
ANNOTATED_FOLDER = os.path.join(OUTPUT_FOLDER, "annotated")

# Setup
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# Initialize Roboflow model
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_NAME)
model = project.version(MODEL_VERSION).model

# Initialize annotator
annotator = sv.BoxAnnotator()

# Inference function
def run_inference(image_path, filename):
    image = cv2.imread(image_path)

    result = model.predict(image_path, confidence=40).json()
    detections = sv.Detections.from_inference(result)

    # Annotate and save image with bounding boxes
    annotated_image = annotator.annotate(scene=image.copy(), detections=detections)
    output_image_path = os.path.join(ANNOTATED_FOLDER, filename)
    cv2.imwrite(output_image_path, annotated_image)

    # Create dataframe from predictions
    records = []
    for pred in result["predictions"]:
        records.append({
            "class": pred["class"],
            "confidence": pred["confidence"],
            "x": pred["x"],
            "y": pred["y"],
            "width": pred["width"],
            "height": pred["height"]
        })

    df = pd.DataFrame(records)

    summary_row = pd.DataFrame([{
        "class": "TOTAL_OBJECTS",
        "confidence": len(records),
        "x": "", "y": "", "width": "", "height": ""
    }])
    df = pd.concat([df, summary_row], ignore_index=True)

    # Save CSV
    output_csv_path = os.path.join(OUTPUT_FOLDER, filename.rsplit(".", 1)[0] + ".csv")
    df.to_csv(output_csv_path, index=False)

    print(f"{filename}: {len(records)} objects detected and saved with annotations")

# Loop through folder
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        run_inference(image_path, filename)
