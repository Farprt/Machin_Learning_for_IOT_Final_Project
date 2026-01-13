# Real-time On-device Gesture Recognition on Arduino

## Project Overview
This project implements a **real-time, fully on-device gesture (spell) recognition system** using IMU time-series data collected from an **Arduino Nano 33 BLE Sense**.

The system transforms the microcontroller into a "Magic Wand" capable of recognizing specific motion patterns (e.g., *Flipendo*, *Wingardium*) and providing immediate feedback via **RGB LEDs** and **Serial output**.

The design emphasizes:
- **Two-stage architectural exploration** (CNN-based Embeddings vs. End-to-End MLP).
- **Physics-based data augmentation** to handle motion variability.
- **Real-time inference** under tight memory constraints.

## System Architecture
This project explored and implemented two distinct model architectures to optimize for on-device performance and few-shot adaptation.

### 1. Explored Framework: Augmented 1D-CNN & On-Device KNN
Initially, a hybrid framework was designed to enable **few-shot personalization** directly on the device.

* **Feature Extractor (1D-CNN):**
    * A **Conv1D backbone** extracts temporal motion patterns from the accelerometer data.
    * Maps raw inputs to a compact **16-dimensional embedding vector**, creating a unique "fingerprint" for each spell.
* **On-Device Decision (KNN):**
    * **Learning Mode:** The device records new gesture embeddings and stores them as prototypes in a local database (up to 10 samples).
    * **Inference Mode:** A **1-Nearest Neighbor (1-NN)** algorithm calculates the Squared Euclidean distance between the live input and stored prototypes to classify the gesture.

### 2. Final Deployed System: End-to-End MLP
To ensure maximum stability and robustness against noise in the final demonstration, the system adopted a lightweight **Multi-Layer Perceptron (MLP)**.

* **Model Structure:** A fully connected network with the architecture **561 → 32 → 16 → 3**.
    * **Input:** Flattened 3-axis accelerometer window (187 steps × 3 axes = 561 features).
    * **Hidden Layers:** 32 and 16 neurons with **ReLU** activation.
    * **Output:** Softmax layer for 3-class classification (*Flipendo*, *Wingardium*, *Others*).
* **Advantages:** This approach proved more robust to variations in user motion compared to the fixed embedding approach and avoided the complexity of managing a dynamic database in RAM.

---

## Data Processing & Augmentation
The dataset was collected via Edge Impulse with a sampling rate of **62.5 Hz** and a recording window of **3 seconds**.

To bridge the gap between limited training samples (300 per class) and complex real-world usage, a **physics-based data augmentation generator** was implemented. It applies six stochastic transformations:

1.  **Axis Permutation:** Simulates 90° grip changes.
2.  **Channel Flip:** Simulates left-handed or backward usage.
3.  **Spatial Rotation:** Applies random 3D rotations (±30°) for wrist tilts.
4.  **Signal Scaling:** Adapts to different casting forces.
5.  **Time Warping:** Handles uneven casting speeds.
6.  **Jitter Injection:** Simulates sensor noise and hand tremors.

---

## Hardware & Deployment
* **Device:** Arduino Nano 33 BLE Sense.
* **Pipeline:**
    1.  **IMU Read:** Continuous 3-axis acceleration monitoring.
    2.  **Gravity Removal:** High-pass filter to isolate linear acceleration (a = raw - g).
    3.  **Trigger Detection:** Recording starts when motion intensity exceeds a threshold.
    4.  **Inference:** The captured window is fed into the TFLite model (MLP) for classification.

## Limitations & Future Work
We identified specific limitations in the current implementation:

* **Temporary Memory:** Patterns learned in the KNN mode are RAM-based and lost upon reset. *Solution: Integrate EEPROM/Flash storage*.
* **Rigid Windowing:** The fixed 3-second window may capture excess noise. *Solution: Adaptive dynamic motion cropping*.
* **Decision Logic:** Simple thresholding can be sensitive to motion intensity. *Solution: Per-user calibration and robust decision rules*.

## Demo Video
A demonstration of the system in action can be found in the presentation silde:

[View Presentation & Demo](./final_project.pptx)
