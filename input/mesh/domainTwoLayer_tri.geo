//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0.1, -0, 0, 1.0};
//+
Point(3) = {0.2, 0, 0, 1.0};
//+
Point(4) = {0.2, 0.1, 0, 1.0};
//+
Point(5) = {0.1, 0.1, 0, 1.0};
//+
Point(6) = {0, 0.1, 0, 1.0};
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
Line(7) = {2, 5};
//+
Curve Loop(1) = {1, 7, 5, 6};
//+
Curve Loop(2) = {2, 3, 4, -7};
//+
Plane Surface(1) = {1};
//+
Plane Surface(2) = {2};
//+
Physical Curve("South") = {1,2};
//+
Physical Curve("East") = {3};
//+
Physical Curve("North") = {4,5};
//+
Physical Curve("West") = {6};
//+
Physical Surface("Domain_1") = {1};
//+
Physical Surface("Domain_2") = {2};
//+
Transfinite Curve {1, 2, 3, 4, 5, 6, 7} = 4 Using Progression 1;
