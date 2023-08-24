# VideoTextExtractor
Project that will analyze and get the regex the user from a video. It allows:
- Use a local video file or a video from YouTube.
- Apply any regex to the OCR extracted text.
- Save the results and frames detected in a folder.
- Select how frequently the frames will be analyzed.
- Set the quality of the video if it is from YouTube. By default it will be 1080p60.

## Requirements
- Python 3.9 or higher
- [Tesseract](https://tesseract-ocr.github.io/tessdoc/Downloads) installed and added to PATH
- Cuda capable GPU (optional)

## Installation
1. Clone the repository
2. Install the requirements
3. Run the program with
```
python .\videoOCR.py -v <video file or yotube uri> -r <regex you want to apply> -o <output folder> -s <every how many frames it will check> -q <quality of the video>
```