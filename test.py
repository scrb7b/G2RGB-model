import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import ColorModel  # Make sure this import matches your model's script

# Load the pre-trained model
model = ColorModel()
model.load_state_dict(torch.load('concatcolor15epochs.pth'))
model.eval()  # Set the model to evaluation mode

# Define the same transformations used in training
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Grayscale transformation
grayscale_transform = transforms.Grayscale()

# Load and preprocess a test image
test_image_path = 'Images/23445819_3a458716c1.jpg'  # Replace with your image path
test_image = Image.open(test_image_path)
input_image = transform(test_image).unsqueeze(0)  # Add batch dimension

# Convert the image to grayscale
grayscale_image = grayscale_transform(input_image)

# Run the grayscale image through the model to get the colorized output
with torch.no_grad():
    colorized_output = model(grayscale_image)

# Denormalize the output to [0, 1] range
colorized_output = colorized_output.squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5

# Display the original, grayscale, and colorized images
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Original image
ax[0].imshow(test_image)
ax[0].set_title('Original Image')
ax[0].axis('off')

# Grayscale image
ax[1].imshow(grayscale_image.squeeze(0).squeeze(0), cmap='gray')
ax[1].set_title('Grayscale Image')
ax[1].axis('off')

# Colorized output
ax[2].imshow(colorized_output)
ax[2].set_title('Colorized Output')
ax[2].axis('off')

plt.show()
