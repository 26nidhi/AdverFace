# AdverFace

**Face Recognition with Adversarial Attack and Detection**

A comprehensive face recognition system demonstrating vulnerabilities in deep learning-based biometric authentication and effective defense mechanisms against adversarial attacks.

## 📋 Overview

AdverFace integrates real-time face recognition using MTCNN and InceptionResnetV1, implements the Reface gradient-based adversarial attack method, and incorporates L2 norm-based detection mechanisms. Through extensive testing, we demonstrate that state-of-the-art face recognition systems achieving 99% accuracy can be completely fooled by imperceptible perturbations with L2 norms below 0.20.

## ✨ Features

- **Real-time Face Recognition**: Live video processing at 10-15 FPS with 95%+ accuracy
- **Adversarial Attack Simulation**: Reface method with 100 optimization iterations
- **Attack Detection**: L2 norm-based detection with real-time security alerts
- **High Performance**: GPU-accelerated processing with graceful degradation
- **Comprehensive Testing**: Demonstrates both vulnerabilities and defenses

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- Webcam (720p minimum, 1080p recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adverface.git
cd adverface
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
facenet-pytorch>=2.5.0
numpy>=1.21.0
Pillow>=8.3.0
```

## 💻 Usage

### Basic Face Recognition

```python
from adverface import FaceRecognition

# Initialize the system
recognizer = FaceRecognition()

# Add known faces
recognizer.add_face("Person1", "path/to/image1.jpg")
recognizer.add_face("Person2", "path/to/image2.jpg")

# Start real-time recognition
recognizer.start_recognition()
```

### Adversarial Attack Generation

```python
from adverface import RefaceAttack

# Initialize attack
attack = RefaceAttack(model, lambda_reg=0.001, learning_rate=0.05)

# Generate adversarial example
adv_image = attack.generate(
    source_image=original_image,
    target_embedding=target_embedding,
    iterations=100
)
```

### Attack Detection

```python
from adverface import L2Detector

# Initialize detector
detector = L2Detector(threshold=0.1)

# Check for adversarial manipulation
is_adversarial, l2_norm = detector.detect(original_image, processed_image)

if is_adversarial:
    print(f"⚠️ Adversarial attack detected! L2 norm: {l2_norm:.4f}")
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   AdverFace System                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│  │  Setup   │ -> │Recognition│ -> │  Attack  │    │
│  │  Phase   │    │   Phase   │    │  Phase   │    │
│  └──────────┘    └──────────┘    └──────────┘    │
│       │               │                │           │
│       v               v                v           │
│  Load Models    Process Video   Generate Adv.     │
│  Create Embeddings  Detect Faces  Examples        │
│                 Match Identities                   │
│                                                     │
│  ┌──────────────────────────────────────────┐     │
│  │          Detection Phase                 │     │
│  │  Compute L2 Norms | Trigger Alerts      │     │
│  └──────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

## 📊 Performance Metrics

### Normal Operation
- **Recognition Accuracy**: 95%+
- **Confidence Scores**: 0.85-0.95 for correct matches
- **False Positive Rate**: <2% at threshold 0.8
- **Processing Latency**: 65-100ms (GPU)
- **Frame Rate**: 10-15 FPS

### Adversarial Attacks
- **Success Rate**: 100% in controlled testing
- **Convergence**: 60-80 iterations
- **L2 Norm**: 0.15-0.20 (imperceptible)
- **Target Similarity**: >0.9

### Detection Performance
- **True Positive Rate**: 100%
- **False Positive Rate**: 0.3%
- **Detection Overhead**: <1ms per frame
- **Real-time Operation**: ✓

## 🔬 Technical Details

### Face Detection (MTCNN)
Three-stage cascaded structure:
1. **P-Net**: Generates initial candidates
2. **R-Net**: Refines detections
3. **O-Net**: Produces final bounding boxes with landmarks

### Face Recognition (InceptionResnetV1)
- **Architecture**: Inception modules + Residual connections
- **Embedding Size**: 512 dimensions
- **Loss Function**: Triplet loss
- **Pre-training**: VGGFace2 (3.3M images, 9,131 identities)

### Reface Attack
Gradient-based optimization targeting specific identities:

```
Loss = -similarity(f(x_adv), e_target) + λ ||x_adv - x_orig||₂
```

Where:
- λ = 0.001 (regularization)
- Learning rate = 0.05
- Iterations = 100

### L2 Detection
Statistical detection based on perturbation magnitude:

```
d = ||x_proc - x_orig||₂ = √(Σ(x_proc,i - x_orig,i)²)
```

Threshold: d > 0.1

## 🖥️ Hardware Requirements

### Minimum
- Intel Core i5 / AMD Ryzen 5 (4+ cores)
- 8GB RAM
- 720p webcam
- 2GB storage

### Recommended
- NVIDIA GTX 1060+ with CUDA support
- 16GB RAM
- 1080p webcam
- 5GB storage

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Normal Recognition Accuracy | 95%+ |
| Attack Success Rate | 100% |
| Detection True Positive Rate | 100% |
| Detection False Positive Rate | 0.3% |
| Average L2 Norm (Attack) | 0.15-0.20 |
| Processing Speed (GPU) | 15-20 FPS |

## ⚠️ Limitations

- Small dataset (2 identities in current implementation)
- Detection can be evaded by adaptive attacks
- White-box attack assumptions
- Focus on digital attacks (not physical-world)
- Computational requirements limit mobile deployment

## 🔮 Future Work

1. **Expand Dataset**: Scale to 50+ identities
2. **Additional Attacks**: Implement FGSM, PGD, C&W
3. **Advanced Detection**: ML-based detectors using autoencoders
4. **Liveness Detection**: Against presentation attacks
5. **Multi-modal Biometrics**: Face + voice + behavioral
6. **Physical Attacks**: Test with printed adversarial examples
7. **Black-box Scenarios**: Evaluate query-based attacks
8. **Security Protocols**: Develop standardized evaluation frameworks

## 📚 References

1. Schroff, F., et al. (2015). FaceNet: A unified embedding for face recognition and clustering. CVPR.
2. Zhang, K., et al. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters.
3. Szegedy, C., et al. (2014). Intriguing properties of neural networks.
4. Goodfellow, I.J., et al. (2015). Explaining and harnessing adversarial examples. ICLR.
5. Madry, A., et al. (2018). Towards deep learning models resistant to adversarial attacks. ICLR.

## 👥 Contributors

**Project Team:**
- Nidhi Walke (2023BCS041)
- Kavya Tantuvay (2023BCS032)
- Ansita Singh (2023BCS075)

**Under the Guidance of:**
- Dr. Narinder Singh Punn

## 🏛️ Institution

**ABV-Indian Institute of Information Technology and Management Gwalior**  
Course: Trustworthy Artificial Intelligence

## 📄 License

This project is part of an academic course at ABV-IIITM Gwalior. For educational and research purposes only.

## 🙏 Acknowledgments

We express sincere gratitude to:
- Dr. Narinder Singh Punn for guidance and mentorship
- ABV-IIITM Gwalior for computational resources
- The open-source community (PyTorch, OpenCV, facenet-pytorch)

## 📧 Contact

For questions or collaboration opportunities, please contact through the institution's official channels.

---

**⚡ Built with PyTorch | 🔒 Advancing Trustworthy AI | 🎓 Academic Research Project**
