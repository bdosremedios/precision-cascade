${prec_casc_dir} = "C:/users/dosre/dev/precision-cascade/"
${matlab_dir} = "C:/users/dosre/dev/MATLAB/"

Write-Output 'Copying MATLAB scripts...'

Copy-Item (Join-Path ${matlab_dir} "create_read_matrices.m") `
          (Join-Path ${prec_casc_dir} "matlab_scripts/create_read_matrices.m")
Copy-Item (Join-Path ${matlab_dir} "create_solve_matrices.m") `
          (Join-Path ${prec_casc_dir} "matlab_scripts/create_solve_matrices.m")
Copy-Item (Join-Path ${matlab_dir} "convert_mat_to_CSV.m") `
          (Join-Path ${prec_casc_dir} "matlab_scripts/convert_mat_to_CSV.m")

Write-Output 'Copying MATLAB generated testing read matrices...'

Copy-Item (Join-Path ${matlab_dir} "read_matrices/*") `
          (Join-Path ${prec_casc_dir} "test/read_matrices/")

Write-Output 'Copying MATLAB generated testing solve matrices...'
          
Copy-Item (Join-Path ${matlab_dir} "solve_matrices/*") `
          (Join-Path ${prec_casc_dir} "test/solve_matrices/")

# Write-Output 'Copying SuiteSparse numerical experiment matrices...'

# Copy-Item (Join-Path ${matlab_dir} "experiment_matrices/*") `
#           (Join-Path ${prec_casc_dir} "test/solve_matrices/")
# Copy-Item ~/win_dosre/dev/MATLAB/experiment_matrices/* ~/dev/numerical_experimentation/data/experiment_matrices/

Write-Output 'Done'