# Multimodal Visual Life Detection using a Compact Tri-Modal Camera Unit - Code Repository

This repository contains the full codebase for the thesis **"Multimodal Visual Life Detection using a Compact Tri-Modal Camera Unit"**, including detection pipelines, evaluation scrips and real-time integration for:

- **Chest Motion Detection** – Life Detection via detecting respiratory motion   
- **Large-Scale Movement Detection** – Life Detection via analysis of victim's movement e.g. walking, crawling, and waving

The code supports both **offline evaluation** and **real-time execution** on the **[Compact Tri-Modal Camera Unit for RGBDT Vision](https://dl.acm.org/doi/fullHtml/10.1145/3523111.3523116)**.

The associated dataset is available on Zenodo:  
[https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Repository Structure

```
.
├── chest-motion-detection/
│   ├── pipeline.py               # Offline evaluation pipeline
│   ├── evaluation.py             # Breathing detection and visualization
│   └── ctcat/
│       ├── live_pipeline.py      # Real-time processing with CTCAT
│       ├── config.yaml           # Config: modalities, ROI, thresholds
│       └── utils/                # CTCAT-specific helper functions
│
├── large-scale-movement-detection/
│   ├── pipeline.py
│   ├── evaluation.py
│   └── ctcat/
│       ├── live_pipeline.py
│       ├── config.yaml
│       └── utils/
│
│── ctcat-live-scripts/
│   ├── large-scale_movement-detection.py
│   ├── chest-motion-detection.py
│   ├── ...utility scripts...
│
├── LICENSE.txt
└── README.md
```

---

## Features

### Chest Motion Detection
- Uses optical flow (Farneback) on rgb, thermal or depth frames
- Extracts motion magnitude from chest ROI
- Visualizes respiratory motion e.g. presence or absence of breathing

### Large-Scale Movement Detection
- Evaluates human movement in indoor/outdoor environments
- Visual overlays of motion 
- Single-person and multi-person scenarios

### Real-Time CTCAT Integration
- Live frame processing using CTCAT sensor
- Configurable ROI and modality selection
- Real-time movement detection (large-scale movement-detection)
- Real-time motion signal plot and arrow overlay visualization (chest motion detection)

---

## Dataset

The dataset used in this repository is publicly available:

 **Multimodal Motion Detection Dataset (Movement & Chest)**  
 [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

Includes:
- Synchronized raw frames (RGB, depth, thermal)
- Scenario-level evaluation outputs (annotations, overlays, breathing curves)

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially

With the requirement that you provide appropriate credit. See `LICENSE.txt` or [creativecommons.org/licenses/by/4.0](https://creativecommons.org/licenses/by/4.0/) for full terms.

---

## Author

**Niklas Grüner**  

