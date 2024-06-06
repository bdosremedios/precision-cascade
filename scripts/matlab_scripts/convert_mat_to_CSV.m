mat_dir = "ssget/mat/HB/";
save_dir = "experiment_matrices/";

files = dir(mat_dir);
files = vertcat({files(:).name})';
files = files(contains(files, ".mat"));

for i = 1:length(files)

    file_name = files(i);
    name = extractBefore(file_name, ".mat");
    mat_file = load(mat_dir + file_name);
    mat_file = mat_file.Problem;
    mat = mat_file.A;
    writematrix(full(mat), save_dir + name + ".csv");

end