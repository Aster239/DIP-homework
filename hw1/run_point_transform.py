import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None


# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()

    if img is None:
        image = None
        return None

    image = np.array(img).copy()
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    return image


# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image

    if image is None:
        return None

    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 4, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 4, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(
            marked_image,
            tuple(points_src[i]),
            tuple(points_dst[i]),
            (0, 255, 0),
            1,
            tipLength=0.2
        )

    return marked_image


def tps_kernel(r2, eps=1e-8):
    return r2 * np.log(r2 + eps)


def solve_tps_parameters(ctrl_pts, values, reg=1e-6):
    n = ctrl_pts.shape[0]

    # K matrix
    diff = ctrl_pts[:, None, :] - ctrl_pts[None, :, :]   # (N, N, 2)
    r2 = np.sum(diff ** 2, axis=2)                       # (N, N)
    K = tps_kernel(r2) + reg * np.eye(n, dtype=np.float64)

    # P matrix for affine term
    P = np.concatenate(
        [np.ones((n, 1), dtype=np.float64), ctrl_pts.astype(np.float64)],
        axis=1
    )  # (N, 3)

    # Full linear system
    L = np.zeros((n + 3, n + 3), dtype=np.float64)
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T

    Y = np.zeros((n + 3,), dtype=np.float64)
    Y[:n] = values.astype(np.float64)

    params = np.linalg.solve(L, Y)
    w = params[:n]
    a = params[n:]
    return w, a


def eval_tps(grid_x, grid_y, ctrl_pts, w, a):
    h, w_img = grid_x.shape
    query = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)  # (HW, 2)

    diff = query[:, None, :] - ctrl_pts[None, :, :]      # (HW, N, 2)
    r2 = np.sum(diff ** 2, axis=2)                       # (HW, N)
    U = tps_kernel(r2)                                   # (HW, N)

    affine = np.concatenate(
        [np.ones((query.shape[0], 1), dtype=np.float64), query],
        axis=1
    )  # (HW, 3)

    values = U @ w + affine @ a
    return values.reshape(h, w_img)


# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    if image is None:
        return None

    warped_image = np.array(image).copy()
    h, w = warped_image.shape[:2]

    n = min(len(source_pts), len(target_pts))
    if n == 0:
        return warped_image

    source_pts = np.asarray(source_pts[:n], dtype=np.float64)
    target_pts = np.asarray(target_pts[:n], dtype=np.float64)

    # Need at least 3 pairs for a stable TPS warp in 2D
    if n < 3:
        return warped_image

    # Inverse warping:
    # For each pixel in the output image, find where to sample in the source image.
    # So we fit the mapping target -> source.
    ctrl_pts = target_pts

    src_x = source_pts[:, 0]
    src_y = source_pts[:, 1]

    wx, ax = solve_tps_parameters(ctrl_pts, src_x, reg=1e-6)
    wy, ay = solve_tps_parameters(ctrl_pts, src_y, reg=1e-6)

    # Dense output grid
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64)
    )

    map_x = eval_tps(grid_x, grid_y, ctrl_pts, wx, ax)
    map_y = eval_tps(grid_x, grid_y, ctrl_pts, wy, ay)

    # Optional strength control
    # alpha=1.0 means exact interpolation
    if abs(alpha - 1.0) > 1e-8:
        map_x = grid_x + alpha * (map_x - grid_x)
        map_y = grid_y + alpha * (map_y - grid_y)

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)

    warped_image = cv2.remap(
        warped_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    return warped_image


def run_warping():
    global points_src, points_dst, image

    if image is None:
        return None

    warped_image = point_guided_deformation(
        image,
        np.array(points_src),
        np.array(points_dst)
    )

    return warped_image


# Clear all selected points
def clear_points():
    global points_src, points_dst, image
    points_src.clear()
    points_dst.clear()
    return image


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("#Facial Expression Warping")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()