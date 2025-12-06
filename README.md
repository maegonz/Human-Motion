# Human Motion Project

## Overview

This project explores the intersection of **Natural Language Processing** and **human motion synthesis**. The project aims to build models that work **bidirectionally**:  

1. **Gesture-to-Motion Generation** – generate 3D human motion from textual descriptions of gestures.  
2. **Motion-to-Text Generation** – generate natural language descriptions from sequences of 3D human motion.  

> **Current focus:** This version of the project implements **Motion-to-Text Generation**. Gesture-to-Motion generation is planned for future work.  

This project is based on a courses conducted by `Hazem Wannous` professor at IMT Nord Europe.
The project uses a dataset containing 3D human motion sequences paired with rich textual descriptions. This enables models to learn mappings between **language** and **motion**.  

---

## Project Roadmap

| Task | Status | Description |
|------|--------|-------------|
| Motion-to-Text Generation | ✅ In Progress / Implemented | Generate natural language descriptions from 3D motion sequences. |
| Gesture-to-Motion Generation | ⏳ Future | Generate 3D motion sequences from textual descriptions of gestures with SMPL models. |

---

<!-- ## Dataset Overview

**HumanML3D** dataset contains:  
- **14,616 motion samples** across actions like walking, dancing, and sports.  
- **44,970 textual annotations**, describing motions in detail.  
- Motion data includes **skeletal joint positions, rotations**, and fine-grained textual descriptions.  

### Data Structure

#### `motions` Folder
- `.npy` files representing sequences of body poses.  
- Shape: `(T, N, d)`  
  - `T`: Number of frames (varies per sequence)  
  - `N`: Number of joints (22)  
  - `d`: Dimension per joint (3D coordinates: x, y, z)  

#### `texts` Folder
- `.npy` files with **3 textual descriptions per motion sequence**  
- Each description includes **part-of-speech (POS) tags**  
- Example:  
"a person jump hop to the right#a/DET person/NOUN jump/NOUN hop/NOUN to/ADP the/DET right/NOUN#" -->
<!-- 

#### File Lists
- `all.txt`: List of all motion files  
- `train.txt`: Training set motion files  
- `val.txt`: Validation set motion files  
- `test.txt`: Test set motion files  

--- -->

<!-- ## Current Usage

The project currently supports:  
- Loading HumanML3D motion and text data  
- Preprocessing 3D motion sequences and textual descriptions  
- Training models for **motion-to-text generation**  

Future updates will include:  
- Gesture-to-motion generation  
- Bidirectional motion-language modeling   -->