$script_dir = $PSScriptRoot

$sub_script = {

    param($parent_script_dir)

    Set-Location $parent_script_dir
    Set-Location ..\..\..

    if (Test-Path -Path build) {
        Remove-Item -Recurse build
    }
    if (Test-Path -Path install) {
        Remove-Item -Recurse install
    }

}

powershell -Command "& {Invoke-Command {$sub_script} -ArgumentList $script_dir}"

exit 0