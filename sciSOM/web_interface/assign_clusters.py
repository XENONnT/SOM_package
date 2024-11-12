import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import argparse

# Function to convert image to base64 for Dash display
def pil_image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64,{}".format(encoded_image)

# Parse command-line argument for initial image
#def parse_args():
#    parser = argparse.ArgumentParser(description="SOM visualization web interface")
#    parser.add_argument('--image', type=str, required=True, help='Path to the initial SOM U-matrix image')
#    return parser.parse_args()

#args = parse_args()

# Load the initial SOM U-matrix (replace with your image)
umatrix_image = Image.open('./figs/4class_2dim_synth_mUmatrix.png')
umatrix_size = umatrix_image.size

# Initialize Dash app
app = dash.Dash(__name__)

# Layout with a default image and an image upload component
app.layout = html.Div([
    html.H1("SOM Interactive Visualization"),
    
    # Display initial U-matrix image
    html.Div([
        html.Img(
            id='umatrix-image', 
            src=pil_image_to_base64(umatrix_image),
            style={'position': 'absolute', 'z-index': 1,
                   'width': '100%', 'height': 'auto'},
            ),  # Call JavaScript function to get image info),
        # Div to hold overlayed image
    html.Img(id='overlay-image', 
             style={'position': 'absolute', 
                    #'top': '100', 'left': '0', 
                    'z-index': 2,
                    'width': '100%', 'height': 'auto'}
                    )
    ],  style={'position': 'relative', 'width': '600px', 'margin': 'auto'}),

    #html.Div(id='trigger', style={'display': 'none'}),

    # Hidden div to store the U-matrix image's position and size
    #html.Div(id='image-info', style={'display': 'none'}),
    
    # Upload another image to overlay
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Image to Overlay')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    
    # Slider to control transparency of overlay
    html.Label('Transparency:'),
    dcc.Slider(
        id='transparency-slider',
        min=0,
        max=1,
        step=0.1,
        value=0.5,  # Default transparency
        marks={i / 10: str(i / 10) for i in range(11)}
    ),
])

# Callback to handle image upload and transparency control
@app.callback(
    Output('overlay-image', 'src'),
    [Input('upload-image', 'contents'), 
     Input('transparency-slider', 'value')]
)
def update_overlay_image(uploaded_image, transparency):
    if uploaded_image is not None:
        # Decode uploaded image
        content_type, content_string = uploaded_image.split(',')
        decoded_image = base64.b64decode(content_string)
        img = Image.open(BytesIO(decoded_image))

        overlay_img = img.resize(umatrix_size, Image.LANCZOS)

        # Convert the overlay image to RGBA and apply transparency
        img = overlay_img.convert("RGBA")
        arr = np.array(img)
        arr[..., 3] = (arr[..., 3] * transparency).astype(np.uint8)  # Adjust transparency
        img = Image.fromarray(arr)

        return pil_image_to_base64(img)

    # Return empty if no image is uploaded
    return None

'''
# Client-side callback to get image position and size using JavaScript
app.clientside_callback(
    """
    function(_, _) {
        const img = document.getElementById('umatrix-image');
        if (img) {
            const rect = img.getBoundingClientRect();
            return JSON.stringify({
                left: rect.left, top: rect.top, width: rect.width, height: rect.height
            });
        }
        return '{}';
    }
    """,
    Output('trigger', 'children'),
    [Input('upload-image', 'contents')]
)

# Adjust overlay image position based on U-matrix image info
@app.callback(
    Output('overlay-image', 'style'),
    [Input('trigger', 'children')]
)
def adjust_overlay_position(image_info):
    if image_info:
        info = eval(image_info)  # Convert JSON string to dictionary
        style = {
            'position': 'absolute',
            'left': f"{info['left']}px",
            'top': f"{info['top']}px",
            'width': f"{info['width']}px",
            'height': f"{info['height']}px",
            'z-index': 2
        }
        return style
    return {'display': 'none'}
'''
    
# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)