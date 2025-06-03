## Crack Analyzer

This project provides a simple Gradio interface for analyzing binary crack images using skeletonization, width measurement, branch detection, and compliance checking.

### Running

Install dependencies:

```
pip install gradio==3.50.2 scikit-image scipy pillow numpy
```

Start the app:

```
python app.py
```

Upload a binary crack image (white=crack, black=background) and adjust the pixel size and compliance thresholds as needed.
