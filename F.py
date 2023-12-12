import os
from requests import get
from random import choice
from threading import Thread
from bs4 import BeautifulSoup
from strgen import StringGenerator
import tkinter as tk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from io import BytesIO

# Additional imports for PyTorch model
from torchvision import models
from torchvision import utils

# PyTorch model setup (assuming you have a pre-trained model)
model = models.resnet50(pretrained=True)
model.eval()

# GUI Setup
root = tk.Tk()
root.title("Image Scraper and Classifier")

# Create images directory
if not os.path.exists('images'):
    os.mkdir('images')

# Save image to file
def save(url):
    file = url.split('/')[-1]
    try:
        data = get(url, allow_redirects=True)
    except:
        pass
    else:
        path = 'images/' + file
        open(path, 'wb').write(data.content)
        if os.path.exists(path):
            print('[+] File ' + file + ' saved. Size: ' + str(os.path.getsize(path)) + ' bytes')
            return path
    return None

# Image classification function using PyTorch
def classify_image(image_path):
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)

    _, predicted_idx = torch.max(output, 1)
    return str(predicted_idx.item())

# Scan function
def scan():
    while True:
        # Random url
        random = StringGenerator('[\h]{6}').render().lower()
        url = 'https://prnt.sc/' + random

        # Make a GET request to get HTML
        content = get(url, timeout=3).text

        # Parse HTML to get page title
        soup = BeautifulSoup(content, 'html.parser')

        # Check if Cloudflare blocked the request
        if 'Cloudflare' in soup.title.get_text().split():
            print('[-] Cloudflare blocked request!')
            break

        # Try to download image
        else:
            try:
                image = soup.img['src']
            except TypeError:
                continue
            else:
                if image.startswith('http'):
                    image_path = save(image)
                    if image_path:
                        classification_result = classify_image(image_path)
                        update_gui(image_path, classification_result)

# Update GUI with image and classification result
def update_gui(image_path, classification_result):
    image = Image.open(image_path)
    image = image.resize((300, 200), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    label_image.config(image=photo)
    label_image.image = photo

    label_result.config(text="Classification Result: " + classification_result)

# GUI elements
label_image = tk.Label(root)
label_image.pack()

label_result = tk.Label(root, text="Classification Result: ")
label_result.pack()

# Start threads
threads_count = 5  # Set the number of threads
for i in range(threads_count):
    thread = Thread(target=scan)
    thread.start()
    print('[*] Starting thread: ' + '[' + str(i) + '/' + str(threads_count) + ']')

# GUI main loop
root.mainloop()
