# VideoLens AI 🎬

**Human-in-the-Loop Video Data Annotation & AI-Generated Caption Quality Assurance**  
*Portfolio project tailored for the micro1 AI/ML Video Data Annotation role.*

---

## Overview

**VideoLens AI** is a production-grade annotation and evaluation prototype built for multimodal AI teams. It pairs video playback and frame inspection with structured human annotation workflows and rigorous claim-by-claim evaluation of AI-generated video descriptions.

Rather than making unsubstantiated claims about automated visual truth, VideoLens AI establishes a disciplined **human-in-the-loop (HITL) protocol**: reviewers ground every annotation and caption claim directly in observable video evidence.

---

## Key Features

1. **Video Ingestion & Frame Inspection**
   - High-performance playback of MP4, MOV, AVI, MKV, and WebM videos.
   - Precise sub-second timestamp frame extraction powered by OpenCV.
   - Windows-optimized video stream caching to eliminate disk thrashing during frame scrubbing.

2. **Structured Video Annotation**
   - Granular labeling across core modalities: **Action**, **Object**, **Person**, **Scene**, and **Audio Event**.
   - Start and end timestamp validation, confidence calibration (Low, Medium, High), and review statuses (`Needs review`, `Reviewed`).
   - Auditable reviewer rationales documenting strictly visible/audible evidence.

3. **AI Description Evaluator (Claim-by-Claim Verification)**
   - Robust NLP sentence and list segmenter splitting captions into verifiable claims.
   - Standardized human verdicts:
     - **Supported**: Observable in video with cited timestamps.
     - **Incorrect**: Directly contradicts video evidence.
     - **Uncertain**: Visual ambiguity, occlusion, or motion blur prevents conclusive verification.
     - **Missing Context**: Partially true but omits critical qualifying details.
   - Built-in video reference player allowing reviewers to verify claims side-by-side with footage.

4. **Annotation Quality Assurance Dashboard**
   - Real-time dataset metrics: total annotations, review coverage percentage, and high-confidence distribution.
   - Label distribution bar charts and an actionable review queue.
   - AI description verdict distributions and audit summaries.

5. **Multi-Format Dataset Exports**
   - JSON export for downstream multimodal model training pipelines.
   - Tabular CSV exports (`annotations.csv` and `description_claims_reviewed.csv`) for QA audits and inter-annotator agreement analysis.

---

## System Architecture & File Structure

```text
VideoLens_AI/
├── app.py                  # Core Streamlit application & processing logic
├── requirements.txt        # Minimal, cross-platform Python dependencies
├── README.md               # Documentation & portfolio overview
├── .gitignore              # Ignores local data, virtual environments, cache
├── tests/
│   └── test_core.py        # 12 automated unit tests (claim parsing, I/O, video decoding)
└── data/
    └── demo_clip.mp4       # Synthetic test video clip for validation
```
Annotation records and description reviews are generated locally when the app is used and are excluded from version control by default.
---

## Quickstart Guide (Windows 10/11)

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13 installed with `Add python.exe to PATH` enabled.

### 2. Environment Setup
Open PowerShell or Command Prompt in the project folder:

```powershell
# Create a virtual environment (optional but recommended)
py -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
# Or in Command Prompt: .venv\Scripts\activate.bat

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Automated Tests
Verify all core modules (parsing, video processing, error recovery):

```powershell
python -m pytest tests -v
```

### 4. Launch VideoLens AI
Start the local Streamlit application:

```powershell
python -m streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

---

## Human Annotation Standard (micro1 Protocol)

1. **Evidence First**: Label only what is directly visible or audible in the clip. Do not infer human intent, identity, or off-screen events.
2. **Tight Temporal Bounds**: Select the narrowest start and end timestamps that completely capture the target event.
3. **Calibrated Uncertainty**: Use the `Uncertain` verdict or `Low` confidence whenever camera perspective or occlusions obscure the event.
4. **Transparent Rationales**: Every non-trivial label or claim verdict must include a reviewer rationale citing timestamp evidence.
5. **Data Privacy**: Annotate only authorized, non-sensitive video clips.

---

## Suggested Next Iterations
- Inter-annotator agreement (IAA) scoring (Cohen's Kappa / Fleiss' Kappa).
- Bounding-box and keypoint spatial annotations.
- Integration with local open-weights Vision-Language Models (e.g., Qwen-VL / LLaVA) for assistive pre-annotation with mandatory human sign-off.
