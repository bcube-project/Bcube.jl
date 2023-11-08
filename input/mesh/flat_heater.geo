cl = 0.006;
//+
Point(1) = {-0, 0, 0, cl};
//+
Point(2) = {0.5, 0, 0, cl};
//+
Point(3) = {0.5, 0.1, 0, cl};
//+
Point(4) = {0.3, 0.1, 0, cl};
//+
Point(5) = {0.3, 0.09, 0, cl};
//+
Point(6) = {0.17, 0.09, 0, cl};
//+
Point(7) = {0.17, 0.1, 0, cl};
//+
Point(8) = {0.3, 0.11, 0, cl};
//+
Point(9) = {0.17, 0.11, 0, cl};
//+
Point(10) = {0.5, 0.17, 0, cl};
//+
Point(11) = {-0, 0.17, 0, cl};
//+
Point(12) = {-0, 0.1, 0, cl};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 12};
//+
Line(8) = {12, 1};
//+
Line(9) = {7, 9};
//+
Line(10) = {9, 8};
//+
Line(11) = {8, 4};
//+
Line(12) = {12, 11};
//+
Line(13) = {11, 10};
//+
Line(14) = {10, 3};
//+
Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
//+
Plane Surface(1) = {1};
//+
Line Loop(2) = {5, 6, 9, 10, 11, 4};
//+
Plane Surface(2) = {2};
//+
Line Loop(3) = {3, -11, -10, -9, 7, 12, 13, 14};
//+
Plane Surface(3) = {3};
//+
Physical Line("FRONT", 1) = {13};
//+
Physical Line("REAR", 2) = {1};
//+
Physical Curve("LEFT", 15) = {2, 14};
//+
Physical Curve("RIGHT", 16) = {8, 12};
//+
Physical Surface("MAT_1", 3) = {1};
//+
Physical Surface("MAT_2", 4) = {3};
//+
Physical Surface("HEATER", 5) = {2};

//+
//Transfinite Surface {1};
//+
//Transfinite Surface {2};
//+
//Transfinite Surface {3};
//+
//Recombine Surface {3};
//+
//Recombine Surface {2};
//+
//Recombine Surface {1};
