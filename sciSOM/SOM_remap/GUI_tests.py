import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Create the image
img_size = (12 * 14) + (13 * 2)  # 12 blocks of size 14 (12 pixels + 2 boundary) and 13 boundaries of size 2
img = np.ones((img_size, img_size, 3))  # Initialized to white

# Display image with matplotlib
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(img)

# Callback for mouse click
def on_click(event):
    # Check which block is clicked and change color
    # ... (This is where most of the logic will be)
    pass

cid = fig.canvas.mpl_connect('button_press_event', on_click)

# Create color picker
color_picker = widgets.ColorPicker(
    value='blue',
    description='Pick a color',
    disabled=False
)
display(color_picker)
