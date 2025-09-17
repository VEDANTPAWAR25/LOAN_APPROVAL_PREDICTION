# Set Python path
$env:PYTHONPATH = $PSScriptRoot
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:CUDA_VISIBLE_DEVICES = "-1"  # Use CPU only, remove if you want to use GPU