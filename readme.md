# Real-time On-device Gesture Recognition on Arduino  
*(Magic Wand Spell Recognition Project)*

## 📌 Project Overview
This project implements a **real-time, fully on-device gesture (spell) recognition system** using IMU time-series data collected from an **Arduino Nano 33 BLE Sense**.

The system recognizes predefined wand gestures (e.g., *Flipendo*, *Wingardium*) and provides immediate feedback via **RGB LEDs** and **Serial output**, without relying on external computation.

The design emphasizes:
- Lightweight on-device learning
- Robustness to user motion variability
- Real-time inference under tight memory constraints

---

## 🧠 System Architecture
The final deployed system adopts an **end-to-end Multi-Layer Perceptron (MLP)** architecture trained and executed entirely on the microcontroller.

### Key characteristics:
- **Input:** 3-axis accelerometer data  
- **Window size:** 187 time steps × 3 axes = 561 features  
- **Model:** End-to-end MLP (561–32–16–3)  
- **Training & inference:** Fully on-device  

Although an **embedding + KNN** pipeline was initially explored for few-shot adaptation,  
the final system uses **MLP-based classification as the primary decision mechanism**.  
A **1-NN (KNN)** method is retained only as an *optional on-device personalization strategy*.

---

## 🔄 Data Processing & Augmentation
To improve robustness and reduce overfitting on a limited dataset, a **physics-based data augmentation strategy** is applied, including:

- Axis permutation  
- Channel flip  
- 3D spatial rotation  
- Signal scaling  
- Time warping  
- Gaussian noise (jitter) injection  

These transformations simulate realistic variations in grip orientation, casting speed, and force intensity.

---


## 🎥 Demo Video
A short demonstration video showing real-time gesture recognition on the Arduino device is available here:

 https://github.com/Farprt/Machin_Learning_for_IOT_Final_Project  


---
