# Extending Neural Networks: Lateral Propagation and Multimodal Convolution

This repository contains the code, experiment outputs, manuscript files, and reproducibility notes for the paper:

**Extending Neural Networks: Lateral Propagation and Multimodal Convolution**

The project investigates two related extensions to neural network architecture:

1. A single-modal lateral-propagation configuration intended to explore localized memory-based learning without backpropagation through time.
2. A multimodal convolutional model that combines visual frame representations and audio-derived spectrogram representations for human activity recognition.

## Repository Structure

```text
.
├── CNN/
│   ├── Data/
│   │   └── human-activity-recognition-video-dataset.zip
│   ├── runs/
│   │   └── multimodal_har_cnn/
│   │       ├── cycle_01_train_500/
│   │       │   ├── classification_report.txt
│   │       │   ├── model.pt
│   │       │   ├── test_predictions.csv
│   │       │   └── test_summary.json
│   │       ├── cycle_02_train_890/
│   │       │   ├── classification_report.txt
│   │       │   ├── model.pt
│   │       │   ├── test_predictions.csv
│   │       │   └── test_summary.json
│   │       ├── cycle_metrics.csv
│   │       ├── dataset_summary.json
│   │       └── final_summary.json
│   ├── Tests/
│   │   └── test.py
│   └── Work/
├── RNN/
│   ├── Data/
│   │   ├── IMDB Dataset.csv
│   │   └── IMDB Dataset.csv.zip
│   ├── Tests/
│   │   └── cnn.py
│   └── Work/
│       └── XF_SQL/
│           ├── sqlite_lateral_rnn_runs.csv
│           └── sqlite_lateral_rnn.sqlite
├── CNNRNN/
│   ├── cnnrnn.tex
│   ├── cnnrnn.pdf
│   └── references.bib
├── lp.tex
├── lp.pdf
├── references.bib
├── wlpeerj.cls
└── README.md
````

## Dataset Information

### Human Activity Recognition Video Dataset

The multimodal convolution experiments use the **Human Activity Recognition (HAR - Video Dataset)** from Kaggle.

* Dataset: Human Activity Recognition (HAR - Video Dataset)
* Authors: Sharjeel M. Rajput, Muhammad Bilal, and Areesha Habib
* Publisher: Kaggle
* Year: 2023
* URL: [https://www.kaggle.com/dsv/5722068](https://www.kaggle.com/dsv/5722068)
* DOI: [https://doi.org/10.34740/KAGGLE/DSV/5722068](https://doi.org/10.34740/KAGGLE/DSV/5722068)

The raw dataset contains approximately 15 GB of video data organized into class-specific directories.

Because of file size constraints, the full raw dataset is not intended to be tracked directly in Git. To reproduce the experiments, download the dataset from Kaggle and place it under:

```text
CNN/Data/
```

Expected local file:

```text
CNN/Data/human-activity-recognition-video-dataset.zip
```

After extraction, the dataset should preserve the original class-directory organization from Kaggle.

### IMDB Dataset

The `RNN/` directory contains earlier lateral-propagation / persistent-memory experiments using the IMDB sentiment dataset.

The IMDB files are included only as supporting experiment materials for the single-modal/lateral-propagation work. They are not required to reproduce the multimodal HAR CNN experiment unless specifically reproducing the `RNN/` experiment path.

## Code Information

### Multimodal CNN

The multimodal CNN experiment is located under:

```text
CNN/Tests/test.py
```

This script performs the multimodal human activity recognition experiment using:

* uniformly sampled RGB video frames
* audio extraction
* log-spectrogram conversion
* independent visual and audio convolutional branches
* late fusion through concatenation
* final MLP classification

Experiment outputs are stored under:

```text
CNN/runs/multimodal_har_cnn/
```

Important output files include:

```text
CNN/runs/multimodal_har_cnn/cycle_metrics.csv
CNN/runs/multimodal_har_cnn/dataset_summary.json
CNN/runs/multimodal_har_cnn/final_summary.json
```

Each training cycle also includes:

```text
classification_report.txt
test_predictions.csv
test_summary.json
model.pt
```

### Single-Modal / Lateral-Propagation Experiments

The `RNN/` directory contains the single-modal lateral-propagation / persistent-memory experiment materials.

Relevant files include:

```text
RNN/Tests/cnn.py
RNN/Work/XF_SQL/sqlite_lateral_rnn_runs.csv
RNN/Work/XF_SQL/sqlite_lateral_rnn.sqlite
```

The CSV file contains recorded run metrics. The SQLite database contains local experimental state and should be treated as a generated/local artifact rather than a required repository dependency.

## Requirements

The experiments were implemented in Python using PyTorch.

Recommended environment:

```text
Python 3.10+
PyTorch
NumPy
pandas
scikit-learn
OpenCV
librosa
soundfile
```

Install dependencies with:

```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python librosa soundfile
```

If a `requirements.txt` file is added, dependencies can be installed with:

```bash
pip install -r requirements.txt
```

## Usage Instructions

### 1. Clone the repository

```bash
git clone https://github.com/kenseitehdev/Extending-Neural-Networks-Lateral-Propagation-and-Multimodal-Convolution.git <REPOSITORY_NAME>
cd <REPOSITORY_NAME>
```

### 2. Download the HAR dataset

Download the dataset from Kaggle:

```text
https://www.kaggle.com/dsv/5722068
```

or DOI:

```text
https://doi.org/10.34740/KAGGLE/DSV/5722068
```

Place the downloaded ZIP file at:

```text
CNN/Data/human-activity-recognition-video-dataset.zip
```

Extract it while preserving the class-directory structure.

### 3. Install requirements

```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python librosa soundfile
```

### 4. Run the multimodal CNN experiment

```bash
python CNN/Tests/test.py
```

The script writes outputs to:

```text
CNN/runs/multimodal_har_cnn/
```

### 5. Review results

Primary result files:

```text
CNN/runs/multimodal_har_cnn/cycle_metrics.csv
CNN/runs/multimodal_har_cnn/final_summary.json
CNN/runs/multimodal_har_cnn/dataset_summary.json
```

Cycle-specific results:

```text
CNN/runs/multimodal_har_cnn/cycle_01_train_500/
CNN/runs/multimodal_har_cnn/cycle_02_train_890/
```

Each cycle directory contains classification reports, predictions, summaries, and trained model checkpoints.

## Methodology

### Data Preprocessing

Videos were split into training and testing sets using an 80/20 split.

For the multimodal CNN experiment, each video was converted into two synchronized modalities:

1. **Visual modality**

   * uniformly sampled RGB frames
   * resized to `128 x 128`
   * normalized to `[0, 1]`

2. **Audio modality**

   * audio extracted at 16 kHz
   * converted into a log-spectrogram
   * resized to `128 x 128`

Training subsets of 500, 1000, and 1500 samples were requested. The maximum usable training size was limited to 890 samples after dataset splitting and class-balanced sampling constraints.

### Multimodal CNN

The multimodal CNN uses two independent convolutional branches:

* a visual branch for sampled RGB frames
* an audio branch for log-spectrograms

The visual branch produces a visual embedding. The audio branch produces an audio embedding. These embeddings are concatenated and passed through a multilayer perceptron classifier.

### Single-Modal Lateral-Propagation Baseline

The single-modal lateral-propagation configuration was evaluated to determine whether localized memory-based updates could capture training-set structure without relying on global temporal backpropagation.

The tested configuration achieved nonzero training accuracy but failed to generalize to held-out samples under the evaluated setup.

## Evaluation Method

The proposed approaches were evaluated using supervised classification experiments.

Training data were used to fit or update each model. Held-out test data were used only for evaluation.

The persistent-memory / lateral-propagation configuration was evaluated by comparing training and test accuracy across sample sizes.

The multimodal CNN was evaluated by comparing held-out test accuracy across available training sizes and against the tested single-modal configuration.

## Assessment Metrics

The primary metrics were:

* **Training accuracy**
* **Test accuracy**

Training accuracy was used to determine whether the model captured structure in the training data.

Test accuracy was used to estimate generalization performance on held-out samples.

For the single-modal lateral-propagation configuration, the gap between training accuracy and test accuracy was used to assess whether the model learned transferable structure or primarily captured training-set-specific patterns.

For the multimodal CNN, changes in test accuracy across training sizes were used to evaluate whether additional data improved generalization.

## Reported Results

### Single-Modal Lateral-Propagation Baseline

| Metric         | 500 Samples | 1000 Samples | 1500 Samples |
| -------------- | ----------: | -----------: | -----------: |
| Train Accuracy |       0.668 |        0.670 |        0.646 |
| Test Accuracy  |       0.000 |        0.000 |        0.000 |

### Multimodal CNN

| Metric         | 500 Samples | 890 Samples |
| -------------- | ----------: | ----------: |
| Train Accuracy |       0.500 |       0.556 |
| Test Accuracy  |       0.413 |       0.637 |

## Manuscript Files

The PeerJ manuscript files are:

```text
lp.tex
lp.pdf
references.bib
wlpeerj.cls
```

The `CNNRNN/` directory contains an earlier related manuscript/preprint version and associated LaTeX build artifacts.

## Reproducibility Notes

The repository provides code, manuscript files, experiment outputs, and reproduction instructions.

Large raw datasets, generated local databases, model checkpoints, and other large artifacts may be excluded from version control. These files can be regenerated or downloaded from the cited dataset source.

Recommended `.gitignore` entries:

```gitignore
# raw datasets and archives
CNN/Data/
RNN/Data/
data/
datasets/
*.zip
*.mp4
*.avi
*.mov
*.mkv

# local databases
*.db
*.sqlite
*.sqlite3

# model checkpoints and generated outputs
*.pt
*.pth
*.ckpt

# generated tables / large metadata
*.tsv
*.parquet
*.feather

# Python cache
__pycache__/
*.pyc
.ipynb_checkpoints/

# LaTeX build artifacts
*.aux
*.bbl
*.blg
*.log
*.out
*.synctex.gz
```

Do not remove final result summaries or manuscript files unless intentionally regenerating them.

## Citation

Dataset citation:

```bibtex
@misc{dataset,
  title     = {Human Activity Recognition (HAR - Video Dataset)},
  author    = {Rajput, Sharjeel M. and Bilal, Muhammad and Habib, Areesha},
  year      = {2023},
  publisher = {Kaggle},
  doi       = {10.34740/KAGGLE/DSV/5722068},
  url       = {https://www.kaggle.com/dsv/5722068}
}
```

## License

No license has been specified yet.

Until a license is added, all rights are reserved by the author. Users should contact the author before reusing code, manuscript text, or experiment materials.

## Contribution Guidelines

This repository is provided primarily for peer-review reproducibility and archival support. External contributions are not currently expected.

For questions about the manuscript or experiments, contact:

```text
Jay Kumar
j.kumar.nbl@gmail.com
```
