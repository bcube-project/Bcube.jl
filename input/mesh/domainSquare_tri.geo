//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, -0, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
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
//+
Transfinite Curve {1, 2, 3, 4} = 4 Using Progression 1;
