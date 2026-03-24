import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication


def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs


def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    if image is None:
        return None

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3),
                         dtype=np.uint8) + np.array((255, 255, 255), dtype=np.uint8).reshape(1, 1, 3)
    image_new[pad_size:pad_size+image.shape[0],
              pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)

    # Get image dimensions and center for transformations
    h, w = image.shape[:2]
    center_x, center_y = w / 2.0, h / 2.0

    # FILL: Apply Composition Transform

    # 1. Flip Matrix (Horizontal flip around the center)
    if flip_horizontal:
        M_flip = np.array([
            [-1,  0, w],
            [0,  1, 0],
            [0,  0, 1]
        ], dtype=np.float32)
    else:
        # Identity matrix if no flip
        M_flip = np.eye(3, dtype=np.float32)

    # 2. Scale and Rotation Matrix (around the center of the image)
    # cv2.getRotationMatrix2D returns a 2x3 matrix, we convert it to 3x3
    M_rot_scale_2x3 = cv2.getRotationMatrix2D(
        (center_x, center_y), rotation, scale)
    M_rot_scale = to_3x3(M_rot_scale_2x3)

    # 3. Translation Matrix
    M_trans = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # Combine all transformations via matrix multiplication (@)
    # Transformation order: Flip -> Rotate & Scale -> Translate
    M_combined = M_trans @ M_rot_scale @ M_flip

    # Extract the final 2x3 affine matrix for OpenCV
    M_final = M_combined[:2, :]

    # Apply the combined affine transformation
    # Using a white border (255, 255, 255) to match your padding color
    transformed_image = cv2.warpAffine(
        image, M_final, (w, h), borderValue=(255, 255, 255))

    return transformed_image

# Gradio Interface


def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")

        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0,
                                  step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(
                    minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(
                    minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(
                    minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")

            # Right: Output image
            image_output = gr.Image(label="Transformed Image")

        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation,
            translation_x, translation_y,
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo


# Launch the Gradio interface
if __name__ == "__main__":
    interactive_transform().launch()
