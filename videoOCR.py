# Extracts the frames from a video file and performs
# OCR on the frames in search of a string that matches a regex
# If a match is found, the frame is saved to a folder and the
# timestamp is saved to a text file with the possible matches

# The script is run with the following command:
# python videoOCR.py -v <video_file> -r <regex> -o <output_folder>

# System imports
import os
import re
import time

# Args parser
import argparse

# OpenCV
# Get cv2 cuda if available
try:
    import cv2.cuda
    cudaDevices = cv2.cuda.getCudaEnabledDeviceCount()
    if cudaDevices == 0:
        # set normal cv2
        del cv2.cuda
        import cv2
except:
    import cv2

# YouTube video capture
from cap_from_youtube import cap_from_youtube

# Tesseract
from PIL import Image
import pytesseract

# TQDM progress bar
from tqdm import tqdm

# Threading
from concurrent.futures import ThreadPoolExecutor

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Extracts frames from a video and performs OCR to search for a regex match.")
    parser.add_argument('-v', '--video', required=True, help="Path to the video file.")
    parser.add_argument('-r', '--regex', required=True, help="Regex pattern to search for.")
    parser.add_argument('-o', '--output', required=True, help="Output folder to save matched frames.")
    parser.add_argument('-s', '--skip', default=0, help="Number of frames to skip between each frame to process.")
    parser.add_argument('-q', '--quality', required=False, default="1080p60", help="Resolution of the video to download from YouTube. Default is 1080.")
    
    return parser.parse_args()

def process_frame(frame, regex, output, timestamp):
    # Perform OCR on the frame
    text = ocr_frame(frame)

    # Match the extracted text to the regex pattern
    matches = match_regex(text, regex)

    # If a match is found, save the frame and timestamp
    if matches:
        save_results(frame, matches, output, timestamp)

# Process the video file
def process_video(video_uri, regex, output, skip_frames, resolution):
    if ("youtube.com" in video_uri or "youtu.be" in video_uri):
        # Get the video capture
        cap = cap_from_youtube(video_uri, resolution=resolution)
    else:
        # Open the video file
        cap = cv2.VideoCapture(video_uri) 

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Iterate through the frames
    with ThreadPoolExecutor() as executor:
        for i in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if necessary
            if i % skip_frames != 0:
                continue

            # Current frame timestamp
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = time.strftime("%H:%M:%S", time.gmtime(timestamp/1000.0))
            
            # Process the frame
            executor.submit(process_frame, frame, regex, output, timestamp)


    # Release the video capture object
    cap.release()

# OCR a single frame
def ocr_frame(frame):
    # Convert the OpenCV frame to a PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform OCR to extract text
    text = pytesseract.image_to_string(image)

    return text

# Match the extracted text to a regex pattern
def match_regex(text, regex_pattern):
    # Search for the regex pattern in the extracted text
    matches = re.findall(regex_pattern, text)
    
    return matches

# Save the results
def save_results(frame, matches, output, timestamp):
    # Create the output folder if it doesn't exist
    os.makedirs(output, exist_ok=True)
    
    # Save the frame as an image
    image_path = os.path.join(output, f"frame_{timestamp}.png")
    cv2.imwrite(image_path, frame)

    # Log the timestamp and matches to a text file
    log_path = os.path.join(output, "matches.txt")
    with open(log_path, 'a') as log_file:
        log_file.write(f"Timestamp: {timestamp}\nMatches: {matches}\n\n")

# Main function
if __name__ == "__main__":
    # Parse the arguments
    args = parse_arguments()

    # Process the video file
    process_video(args.video, args.regex, args.output, int(args.skip), args.quality)