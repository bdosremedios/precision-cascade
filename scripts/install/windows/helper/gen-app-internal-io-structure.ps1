if (-not (Test-Path -Path benchmark)) {
    $null = New-Item benchmark -ItemType Directory
}
if (-not (Test-Path -Path benchmark\output_data)) {
    $null = New-Item benchmark\output_data -ItemType Directory
}

if (-not (Test-Path -Path experimentation)) {
    $null = New-Item experimentation -ItemType Directory
}

if (-not (Test-Path -Path experimentation\matrix_data)) {
    $null = New-Item experimentation\matrix_data -ItemType Directory
}
if (-not (Test-Path -Path experimentation\input_specs)) {
    $null = New-Item experimentation\input_specs -ItemType Directory
}
if (-not (Test-Path -Path experimentation\output_data)) {
    $null = New-Item experimentation\output_data -ItemType Directory
}

exit 0