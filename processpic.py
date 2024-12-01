from PIL import Image
import numpy as np
import csv

# Open and process the image
img = Image.open("letter_4.jpeg")
img = img.convert("L")  # Convert to grayscale
img = img.resize((28, 28))  # Resize to 28x28 pixels

# Convert the image to a NumPy array
array = np.array(img)

# Transpose the array to switch rows and columns
array = array.T

# Flatten the transposed array to a single row
flattened_pixels = array.flatten()

# Define the class value
# 'N' has ASCII value 78. Subtract 48 to get the class value.
class_value = ord('V') - 48

# Prepare the row: [class_value, pixel1, pixel2, ..., pixel784]
row = [class_value] + flattened_pixels.tolist()

# Append the row to the CSV file
with open(file="data/custom-letter.csv", mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(row)

print("Row appended successfully!")