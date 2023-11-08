c1 =  0.01;
//+
Point(1) = {0, 0, 0, c1};
//+
Point(2) = {1., -0, 0, c1};
//+
Point(3) = {1., 0.1, 0, c1};
//+
Point(4) = {1.0, 0.2, 0, c1};
//+
Point(5) = {0.0, 0.2, 0, c1};
//+
Point(6) = {0, 0.1, 0, c1};
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
Line(6) = {6, 1};
//+
Curve Loop(1) = {1, 2, 3, 4, 5, 6};
//+
Plane Surface(1) = {1};
//+
Physical Curve("South") = {1};
//+
Physical Curve("East1") = {2};
//+
Physical Curve("East2") = {3};
//+
Physical Curve("North") = {4};
//+
Physical Curve("West2") = {5};
//+
Physical Curve("West1") = {6};
//+
Physical Surface("Domain") = {1};

