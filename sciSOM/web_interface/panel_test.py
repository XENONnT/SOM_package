import panel as pn
import matplotlib.pyplot as plt
import numpy as np

# Enable Panel extensions
pn.extension()

# Generate a sample SOM U-matrix as an example
fig, ax = plt.subplots(figsize=(6, 6))
data = np.random.rand(10, 10)  # Example U-matrix data
cax = ax.matshow(data, cmap='viridis')
fig.colorbar(cax)

# Create a Panel widget for the Matplotlib plot
som_plot = pn.pane.Matplotlib(fig, width=500, height=500)

# Add a color selection widget to allow users to change colors
color_selector = pn.widgets.ColorPicker(name='Select Color', value='#0000FF')

# Define an interactive callback to update colors (you can expand this logic)
@pn.depends(color_selector)
def update_color(color):
    ax.matshow(data, cmap='cool')  # Example: Changing the color map dynamically
    fig.canvas.draw()
    return som_plot

# Layout the Panel dashboard
dashboard = pn.Column(
    som_plot,
    color_selector,
    pn.bind(update_color, color_selector)
)

# Serve the Panel dashboard
dashboard.show()