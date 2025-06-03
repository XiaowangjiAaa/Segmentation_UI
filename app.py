import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import morphology, draw
import gradio as gr


def load_image(image):
    gray = image.convert("L")
    arr = np.array(gray)
    binary = arr > 128
    return binary


def skeletonize_image(binary):
    skeleton = morphology.thin(binary)
    return skeleton


def find_contour(binary):
    eroded = ndi.binary_erosion(binary)
    contour = binary ^ eroded
    points = np.column_stack(np.nonzero(contour))
    return points


def max_crack_width(binary, skeleton):
    contour_pts = find_contour(binary)
    tree = cKDTree(contour_pts)
    skeleton_pts = np.column_stack(np.nonzero(skeleton))
    max_width = 0.0
    max_pair = None
    for pt in skeleton_pts:
        dists, idxs = tree.query(pt, k=2)
        if len(idxs) < 2:
            continue
        p1 = contour_pts[idxs[0]]
        p2 = contour_pts[idxs[1]]
        width = np.linalg.norm(p1 - p2)
        if width > max_width:
            max_width = width
            max_pair = (p1, p2)
    return max_width, max_pair


def width_visualization(binary, max_pair):
    img = Image.fromarray((binary * 255).astype(np.uint8)).convert("RGB")
    if max_pair is not None:
        draw_img = ImageDraw.Draw(img)
        p1 = tuple(max_pair[0][::-1])
        p2 = tuple(max_pair[1][::-1])
        draw_img.line([p1, p2], fill="yellow", width=1)
    return img


def branch_and_endpoints(skeleton):
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neighbor_count = ndi.convolve(skeleton.astype(int), kernel, mode="constant", cval=0)
    endpoints = (skeleton & (neighbor_count == 1))
    branchpoints = (skeleton & (neighbor_count >= 3))
    overlay = Image.fromarray((skeleton * 255).astype(np.uint8)).convert("RGB")
    draw_img = ImageDraw.Draw(overlay)
    for y, x in np.column_stack(np.nonzero(endpoints)):
        draw_img.ellipse((x - 1, y - 1, x + 1, y + 1), fill="green")
    for y, x in np.column_stack(np.nonzero(branchpoints)):
        draw_img.ellipse((x - 1, y - 1, x + 1, y + 1), fill="red")
    return overlay, endpoints.sum(), branchpoints.sum()


def compute_metrics(binary, skeleton, max_width_px, pixel_size):
    pixel_area = pixel_size ** 2
    area = binary.sum() * pixel_area
    length = skeleton.sum() * pixel_size
    avg_width = area / length if length > 0 else 0
    max_width = max_width_px * pixel_size
    return area, length, avg_width, max_width


def analyze(image, pixel_size, max_width_thresh, avg_width_thresh, area_ratio_thresh, length_thresh):
    binary = load_image(image)
    skeleton = skeletonize_image(binary)
    max_width_px, max_pair = max_crack_width(binary, skeleton)
    width_img = width_visualization(binary, max_pair)
    overlay, endpoint_count, branch_count = branch_and_endpoints(skeleton)
    area, length, avg_width, max_width = compute_metrics(binary, skeleton, max_width_px, pixel_size)
    total_area = binary.size * (pixel_size ** 2)
    area_ratio = (area / total_area) * 100
    est_branch_count = max(branch_count - 1, 0)
    compliance = (
        (max_width <= max_width_thresh) and
        (avg_width <= avg_width_thresh) and
        (area_ratio <= area_ratio_thresh) and
        (length <= length_thresh)
    )
    metrics = {
        "Area (mm^2)": area,
        "Length (mm)": length,
        "Average width (mm)": avg_width,
        "Maximum width (mm)": max_width,
        "Endpoint count": int(endpoint_count),
        "Branch point count": int(branch_count),
        "Estimated branch count": int(est_branch_count),
        "Area ratio (%)": area_ratio,
    }
    return (
        Image.fromarray((binary * 255).astype(np.uint8)),
        Image.fromarray((skeleton * 255).astype(np.uint8)),
        overlay,
        width_img,
        metrics,
        "Pass" if compliance else "Fail",
    )


def main():
    with gr.Blocks(title="Crack Analyzer") as demo:
        gr.Markdown("# Crack Analyzer")
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Binary Crack Image")
                pixel_size = gr.Number(value=1.0, label="Pixel size (mm)")
                max_width_thresh = gr.Number(value=1.0, label="Max width limit (mm)")
                avg_width_thresh = gr.Number(value=1.0, label="Avg width limit (mm)")
                area_ratio_thresh = gr.Number(value=100.0, label="Area ratio limit (%)")
                length_thresh = gr.Number(value=100.0, label="Crack length limit (mm)")
                analyze_btn = gr.Button("Analyze")
            with gr.Column():
                orig = gr.Image(label="Original")
                skel = gr.Image(label="Skeleton")
                overlay = gr.Image(label="Branches & Endpoints")
                width_viz = gr.Image(label="Max Width")
        metrics = gr.JSON(label="Metrics")
        compliance = gr.Textbox(label="Compliance")
        analyze_btn.click(
            analyze,
            inputs=[img_in, pixel_size, max_width_thresh, avg_width_thresh, area_ratio_thresh, length_thresh],
            outputs=[orig, skel, overlay, width_viz, metrics, compliance],
        )
    demo.launch()


if __name__ == "__main__":
    main()
