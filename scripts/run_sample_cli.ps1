param(
  [string]$Image = "data/raw/Patterns/0_0000_00001.jpg",
  [string]$Model = "models/cnn_latest.pth",
  [string]$Device = "auto"
)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  python -m venv .venv
}

.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python -m app.main --image $Image --model $Model --device $Device


