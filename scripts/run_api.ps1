param(
  [string]$Host = "0.0.0.0",
  [int]$Port = 5000
)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  python -m venv .venv
}

.venv\Scripts\python -m pip install -r requirements.txt

$env:FLASK_RUN_HOST = $Host
$env:FLASK_RUN_PORT = $Port

.venv\Scripts\python -m app.flask_app


