square1 = [
    1.0, 2.0, 3.0;
    4.0, 5.0, 6.0;
    7.0, 8.0, 9.0
];
writematrix(square1, "read_matrices\\square1.csv");

square2 = [
    1, 2, 3, 4, 5;
    6, 7, 8, 9, 10;
    11, 12, 13, 14, 15;
];
writematrix(square2, "read_matrices\\square2.csv");

wide = [
    10, 9, 8, 7, 6;
    5, 4, 3, 2, 1;
];
writematrix(wide, "read_matrices\\wide.csv");

tall = [
    1, 2;
    3, 4;
    5, 6;
    7, 8
];
writematrix(tall, "read_matrices\\tall.csv");

empty = [];
writematrix(empty, "read_matrices\\empty.csv");

precise = [exp(1), exp(1)+1;
           exp(1)+2, exp(1)+3];
writematrix(precise, "read_matrices\\precise.csv");
