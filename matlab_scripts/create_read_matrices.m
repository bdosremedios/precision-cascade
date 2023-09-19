vector = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0];
writematrix(vector, "read_matrices\\vector.csv");

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
    16, 17, 18, 19, 20;
    21, 22, 23, 24, 25;
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

half_precise = [
    1.123, 1.124;
    1.125, 1.126
];
writematrix(half_precise, "read_matrices\\half_precise.csv");

single_precise = [
    1.12345672, 1.12345674;
    1.12345676, 1.12345678
];
writematrix(single_precise, "read_matrices\\single_precise.csv");

double_precise = [
    1.12345678901232, 1.12345678901234;
    1.12345678901236, 1.12345678901238
];
writematrix(double_precise, "read_matrices\\double_precise.csv");
