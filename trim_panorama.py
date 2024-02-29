from wand.image import Image
from wand.color import Color
import numpy as np
import cv2
import os

# Path to the stitched image
stitched_image_path = 'uploads/final_panorama/final_panorama.jpg'

# Read the stitched image
stitched = cv2.imread(stitched_image_path)

# Trim the image
stitched2 = stitched.copy()
stitched2 = Image.from_array(stitched2)
stitched2.trim(color=Color('rgb(0,0,0)'), percent_background=0.0, fuzz=0)
stitched2 = np.array(stitched2)

# Save the trimmed image
output_folder = 'uploads/final_panorama'
output_path = os.path.join(output_folder, "final_panorama_trimmed.jpg")
cv2.imwrite(output_path, stitched2)

print(f"Stitched and trimmed image saved at: {output_path}")
