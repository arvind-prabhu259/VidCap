# VidCap
This project was created as part of the Computer Vision course.
It aims to create accurate video captions using video data.

# Dataset
MSVD dataset is used. The dataset consists of links to short Youtube videos and captions. The videos were extracted from youtube and used to train the model.

# Architecture

# VidCap: Video Captioning System
This project was developed as part of a Computer Vision course. It focuses on generating accurate and descriptive captions for video content using deep learning techniques. The project combines video frame feature extraction and sequence-to-sequence modeling to generate captions for short videos.

# Dataset: MSVD (Microsoft Video Description Dataset)
The MSVD dataset is used for training and evaluation. It contains short YouTube videos with associated human-written captions. For this project:

Video links from the MSVD dataset were used to download the videos.
The downloaded videos were processed to extract frames, which were used for training and evaluation.
# Implementation Details
## Frameworks and Tools
Programming Language: Python
Deep Learning Framework: Keras with TensorFlow backend
# System Architecture
## 1. Video Processing
### Frame Extraction:
80 frames are extracted from each video, ensuring uniform representation.
### Feature Extraction:
A pretrained VGG16 model is used to process each frame.
Features from the penultimate layer (a 4096-dimensional vector) are extracted for each frame.
These features are stored in an array of shape (80, 4096) for each video.
## 2. Caption Generation
# Model Architecture:
![image](https://github.com/user-attachments/assets/c4fc5893-1ffa-4283-974b-273d81616efd)
Encoder: A stack of Long Short-Term Memory (LSTM) layers processes the sequential frame features extracted from VGG16.
Decoder: Another LSTM network generates captions, conditioned on the output of the encoder.
## Input to the Model:
Video features: (80, 4096) array from the encoder.
Word embeddings for tokenized captions.
## Output: A sequence of words forming a caption for the input video.
# Training and Optimization
Loss Function: Categorical cross-entropy was used for optimizing the model during training.
Optimization Algorithm: Adam optimizer, with appropriate learning rate tuning.
Training Data: A subset of the MSVD dataset, processed for both video frames and corresponding tokenized captions.
# Results
The performance of the model was evaluated using standard metrics for text generation tasks:
BLEU-4 (Bilingual Evaluation Understudy Score): Measures n-gram precision with a brevity penalty to account for overly short generated captions.
Achieved BLEU-4 score: 0.151
METEOR (Metric for Evaluation of Translation with Explicit ORdering): Evaluates caption quality based on precision, recall, and synonym matching.
Achieved METEOR score: 0.043
These scores reflect the model's ability to generate captions that partially align with human-written descriptions.

Future Improvements
Data Augmentation: Increase the diversity of training data by incorporating additional datasets or augmenting the MSVD dataset.
Attention Mechanism: Incorporate attention layers to focus on relevant frames when generating captions.
Transformer Models: Explore transformer-based architectures (e.g., Vision Transformers and GPT) for improved sequence generation.
Evaluation Metrics: Consider additional evaluation metrics like CIDEr and ROUGE for more comprehensive analysis.
Conclusion
The VidCap project demonstrates a promising approach to video captioning by combining deep learning techniques with feature extraction from pretrained models. While the current implementation provides a good starting point, there is significant scope for enhancements in both model design and training methodology.
