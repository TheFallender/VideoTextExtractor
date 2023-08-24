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


# Class for the options for the program
class VideoOCROptions:
    def __init__(self, video_uri, regex, output, skip_frames, resolution, no_image):
        self.video_uri = video_uri
        self.regex = regex
        self.output = output
        self.skip_frames = skip_frames
        self.resolution = resolution
        self.no_image = no_image


# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extracts frames from a video and performs OCR to search for a regex match.")
    parser.add_argument('-v', '--video', required=True,
                        help="Path to the video file.")
    parser.add_argument('-r', '--regex', required=True,
                        help="Regex pattern to search for.")
    parser.add_argument('-o', '--output', required=True,
                        help="Output folder to save matched frames.")
    parser.add_argument('-s', '--skip', default=0,
                        help="Number of frames to skip between each frame to process.")
    parser.add_argument('-q', '--quality', required=False, default="1080p60",
                        help="Resolution of the video to download from YouTube. Default is 1080.")
    parser.add_argument('--no-image', action='store_true', required=False, default=False,
                        help="Do not save the matched frames as images.")

    return parser.parse_args()


def process_frame(frame, timestamp, options: VideoOCROptions):
    # Perform OCR on the frame
    text = ocr_frame(frame)

    # Match the extracted text to the regex pattern
    matches = match_regex(text, options.regex)

    # If a match is found, save the frame and timestamp
    if matches:
        save_results(frame, matches, timestamp, options)


# Process the video file
def process_video(options: VideoOCROptions):
    if "youtube.com" in options.video_uri or "youtu.be" in options.video_uri:
        # Get the video capture
        cap = cap_from_youtube(options.video_uri, resolution=options.resolution)
    else:
        # Open the video file
        cap = cv2.VideoCapture(options.video_uri)

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
            if i % options.skip_frames != 0:
                continue

            # Current frame timestamp
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Extract the whole seconds and the remaining milliseconds
            seconds = int(timestamp_ms // 1000)
            milliseconds = int(timestamp_ms % 1000)

            # Format the seconds into HH:MM:SS and append the milliseconds
            timestamp = time.strftime("%H:%M:%S", time.gmtime(seconds)) + f".{milliseconds:03}"

            # Process the frame
            executor.submit(process_frame, frame, timestamp, options)

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
def save_results(frame, matches, timestamp, options: VideoOCROptions):
    # Create the output folder if it doesn't exist
    os.makedirs(options.output, exist_ok=True)

    # Save the frame as an image
    image_path = os.path.join(options.output, f"frame_{timestamp.replace(':','')}.png")
    cv2.imwrite(image_path, frame)

    # Log the timestamp and matches to a text file
    log_path = os.path.join(options.output, "matches.txt")
    with open(log_path, 'a') as log_file:
        log_file.write(f"Timestamp: {timestamp}\nMatches: {matches}\n\n")


# Main function
if __name__ == "__main__":
    # Parse the arguments
    args = parse_arguments()

    # Create the options object
    video_ocr_options = VideoOCROptions(
        args.video,
        args.regex,
        args.output,
        int(args.skip),
        args.quality,
        args.no_image
    )

    # Process the video file
    process_video(video_ocr_options)
