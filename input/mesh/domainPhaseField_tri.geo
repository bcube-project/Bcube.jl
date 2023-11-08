c1 =  0.075;
//+
Point(1) = {0, 0, 0, c1};
//+
Point(2) = {12., -0, 0, c1};
//+
Point(3) = {12.0, 3.0, 0, c1};
//+
Point(4) = {0, 3.0, 0, c1};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Curve("South") = {1};
//+
Physical Curve("East") = {2};
//+
Physical Curve("North") = {3};
//+
Physical Curve("West") = {4};
//+
Physical Surface("Domain") = {1};

