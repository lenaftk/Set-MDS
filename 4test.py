import cv2
import os

# Set the directory where your PNG files are located
png_dir = './savefigs'

# Set the name of the output video file
video_name = 'output.mp4'

# Set the frame rate for the output video (in frames per second)
fps = 30

# Get a list of all the PNG files in the directory, sorted by modification time
png_files = sorted([os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith('.png')], key=os.path.getmtime)


# Read the first image to get the dimensions of the images
img = cv2.imread(png_files[0])
height, width, channels = img.shape

# Create the video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Loop through all the PNG files and add them to the video
for png_file in png_files:
    img = cv2.imread(png_file)
    out.write(img)

# Release the video writer object and close the video file
out.release()