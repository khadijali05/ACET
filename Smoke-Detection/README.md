# Smoke-Detection

Professional Python application for detecting vehicle smoke in CCTV streams and uploaded videos. It uses the existing local YOLO checkpoints, GPU inference, temporary video testing, encrypted camera credentials, SQLite event storage, CSV exports, and an administrator dashboard.

## Requirements

- Windows 10 or 11
- Python 3.10
- NVIDIA GPU and a working NVIDIA driver
- Internet access on the first Florence-2 model load

The verified environment uses PyTorch 2.14.0 with CUDA 13.0 (`cu130`). The `requirements.txt` file requests that GPU build.

Florence-2 also requires `einops`, `timm`, and `accelerate`; all three are included because the existing colour-classification model imports or uses them while loading.

`lap` is included because the configured vehicle tracker uses it for track assignment.

## Optional runtime settings

The application uses `cuda:0` by default, which selects the first NVIDIA GPU. You can adjust the GPU device or processing values for the current PowerShell window before starting the dashboard:

```powershell
$env:SMOKE_DETECTION_DEVICE = "cuda:0"
$env:SMOKE_DETECTION_SMOKE_CONFIDENCE = "0.50"
$env:SMOKE_DETECTION_VEHICLE_CONFIDENCE = "0.25"
$env:SMOKE_DETECTION_ANALYSIS_FPS = "5"
```

All values are validated when the dashboard starts. The Settings page shows the active values.

## Create the project environment

Open PowerShell in this folder and run:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check that PyTorch can see the NVIDIA GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU not available')"
```

The first value should be `True`. If it is `False`, check the NVIDIA driver and reinstall the CUDA-enabled PyTorch build shown in `requirements.txt`.

## Configure protected settings

The dashboard requires an administrator username, a hashed administrator password, and a Fernet key for saved camera credentials. Do not put real values in source files.

Run the secure local setup helper once. It prompts for your password without displaying it, saves only a password hash, and generates the camera encryption key:

```powershell
python setup_security.py
```

Close PowerShell and open a new one after the helper finishes. The new terminal will receive the saved Windows user environment settings.

## Start the dashboard

With the virtual environment active and the settings configured:

```powershell
streamlit run dashboard.py
```

Open the local address Streamlit prints, sign in, and add the real camera stream addresses. Enter camera usernames and passwords in their own protected fields, never in the stream address.

## Email alerts

Open **Settings** in the dashboard and complete **Email detection alerts** when you are ready. Enter your recipient email address, SMTP server details, sender email, and SMTP app password. The password is encrypted locally with the key created by `setup_security.py`; it is never shown in the dashboard.

For a Gmail address, use `smtp.gmail.com`, port `587`, STARTTLS, and a Google App Password. Save the settings, send the built-in test email, then enable alerts. Each saved CCTV detection sends an email with its camera name, UTC time, smoke type and confidence, vehicle association and track ID, plus its plate result.

## Model files

The project uses these existing copied checkpoints in `models`:

- `smoke_detector.pt` — smoke detection
- `vehicle_detector.pt` — vehicle detection and tracking
- `plate_detector.pt` — locates license-plate regions on associated vehicles

The smoke colour classifier uses the existing Florence-2 model identifier (`microsoft/Florence-2-base`). Its files download to `models/florence2_cache` the first time it is used. It classifies black, blue, white, and grey smoke when the available evidence supports a classification; otherwise it reports uncertainty or unavailability.

## How live monitoring works

1. You add real CCTV stream settings in the Cameras page.
2. You start monitoring from Live monitoring.
3. The application loads the smoke and vehicle checkpoints once on the GPU.
4. Each enabled camera has its own tracker state. Smoke is associated with a vehicle only when the geometric evidence is sufficient.
5. Real CCTV events are saved in `data/events.sqlite3`, then appear in Detection events and Reports.

Uploaded video tests are temporary. They are processed and shown in the dashboard but are not saved as CCTV events.

## Plate information

The system uses `plate_detector.pt` to locate the plate region on an associated vehicle, then uses the approved EasyOCR component on the GPU to read the plate text. EasyOCR downloads its required OCR weights into `models/easyocr_cache` the first time it is used.

Plate text is saved only when OCR returns a usable result. When the plate is too small, blurry, obstructed, or not readable, the event reports its plate location status without inventing text.

## Important limitations

Smoke-to-vehicle association is an evidence-based geometric estimate. A smoke detection alone does not prove which vehicle emitted it. Review the association status and score for each event.

