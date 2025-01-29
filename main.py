import load_image,edge_detection from image_utils.py

def save_image(img, path):
    img = Image.fromarray(img)
    img.save(path)

path_to_image = "./image.jpg"

# Load the image
img = load_image(path_to_image)

# Perform edge detection
edges = edge_detection(img)
path_to_save = "./output.png"
# Save the image
save_image(edges, path_to_save)
