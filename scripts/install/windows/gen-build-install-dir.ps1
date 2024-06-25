$script_dir = $PSScriptRoot

$sub_script = {

    param($parent_script_dir)

    Set-Location $parent_script_dir
    Set-Location ..\..\..

    if (-not (Test-Path -Path build)) {
        $null = New-Item build -ItemType Directory
    }
    if (-not (Test-Path -Path install)) {
        $null = New-Item install -ItemType Directory
    }

    Set-Location install

    if (-not (Test-Path -Path test)) {
        $null = New-Item test -ItemType Directory
    }
    if (-not (Test-Path -Path test\data)) {
        $null = New-Item test\data -ItemType Directory
    }
    if (-not (Test-Path -Path test\data\read_matrices)) {
        $null = New-Item test\data\read_matrices -ItemType Directory
    }
    Copy-Item ..\test\data\read_matrices\* .\test\data\read_matrices
    if (-not (Test-Path -Path test\data\solve_matrices)) {
        $null = New-Item test\data\solve_matrices -ItemType Directory
    }
    Copy-Item ..\test\data\solve_matrices\* .\test\data\solve_matrices

    ../scripts/install/windows/helper/gen-app-internal-io-structure.ps1

    if (-not (Test-Path -Path experimentation\test)) {
        $null = New-Item experimentation\test -ItemType Directory
    }
    if (-not (Test-Path -Path experimentation\test\data)) {
        $null = New-Item experimentation\test\data -ItemType Directory
    }
    if (-not (Test-Path -Path experimentation\test\data\test_data)) {
        $null = New-Item experimentation\test\data\test_data `
                -ItemType Directory
    }
    Copy-Item ..\experimentation\test\data\test_data\* `
              .\experimentation\test\data\test_data
    if (-not (Test-Path -Path experimentation\test\data\test_jsons)) {
        $null = New-Item experimentation\test\data\test_jsons `
                -ItemType Directory
    }
    Copy-Item ..\experimentation\test\data\test_jsons\* `
              .\experimentation\test\data\test_jsons
    if (-not (Test-Path -Path experimentation\test\data\test_output)) {
        $null = New-Item experimentation\test\data\test_output `
                -ItemType Directory
    }

}

powershell -Command "& {Invoke-Command {$sub_script} -ArgumentList $script_dir}"

exit 0