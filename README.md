## Crack Analyzer

This project provides a Gradio-based interface for inspecting binary crack images (white = crack, black = background).
It extracts a skeleton of the crack network, finds the maximum crack width, detects branch/end points, computes geometric
features, and checks compliance against user defined limits.

### Features
* **Skeleton Extraction** using `skimage.morphology.thin`.
* **Maximum Crack Width** located with a KD-tree and visualized on the image.
* **Branch and Endpoint Detection** via a 3×3 convolution overlaying green (end) and red (branch) dots.
* **Geometric Metrics** including area, length, average width, maximum width, endpoint count, branch point count and an estimated branch count.
* **Compliance Checking** against limits for maximum/average width, area ratio and crack length.

### Running

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the app:
```bash
python app.py
```

Upload a binary crack image and adjust the pixel size and compliance thresholds as needed. The app will display the
original image, the extracted skeleton, branch/endpoint overlay and maximum width visualization along with computed metrics.
