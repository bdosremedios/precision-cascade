$script_dir = $PSScriptRoot

$sub_script = {

    param($parent_script_dir)

    Set-Location $parent_script_dir
    Set-Location ..\..\..
    
    .\scripts\install\windows\gen-build-install-dir.ps1

    Set-Location build
    cmake ..
    msbuild ./precision-cascade.sln
    Set-Location ..
    
    if (Test-Path -Path install\test\test.exe) {
        Remove-Item install\test\test.exe
    }
    Move-Item build\test\Debug\test.exe install\test\test.exe

    if (Test-Path -Path install\benchmark\benchmark.exe) {
        Remove-Item install\benchmark\benchmark.exe
    }
    Move-Item build\benchmark\Debug\benchmark.exe `
              install\benchmark\benchmark.exe

    if (Test-Path -Path install\experimentation\experiment.exe) {
        Remove-Item install\experimentation\experiment.exe
    }
    Move-Item build\experimentation\main\Debug\experiment.exe `
              install\experimentation\experiment.exe

    if (Test-Path -Path install\experimentation\test\test_experiment.exe) {
        Remove-Item install\experimentation\test\test_experiment.exe
    }
    Move-Item build\experimentation\test\Debug\test_experiment.exe `
              install\experimentation\test\test_experiment.exe

}

powershell -Command "& {Invoke-Command {$sub_script} -ArgumentList $script_dir}"

exit 0