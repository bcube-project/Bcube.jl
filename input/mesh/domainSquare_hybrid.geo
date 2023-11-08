//  F----E----D
//  |         |
//  |         |
//  A----B----C

// Points
A = newp; Point(A) = {-1, -1, 0};
B = newp; Point(B) = { 0, -1, 0};
C = newp; Point(C) = { 1, -1, 0};
D = newp; Point(D) = { 1,  1, 0};
E = newp; Point(E) = { 0,  1, 0};
F = newp; Point(F) = {-1,  1, 0};

// Lines
AB = newl; Line(AB) = {A, B};
BC = newl; Line(BC) = {B, C};
CD = newl; Line(CD) = {C, D};
DE = newl; Line(DE) = {D, E};
EF = newl; Line(EF) = {E, F};
FA = newl; Line(FA) = {F, A};
BE = newl; Line(BE) = {B, E};

// Line loops
ABEF = newll; Line Loop(ABEF) = {AB, BE, EF, FA};
BCDE = newll; Line Loop(BCDE) = {BC, CD, DE, -BE};

// Surfaces
left = news; Plane Surface(left) = {ABEF};
right = news; Plane Surface(right) = {BCDE};

// Physical groups
Physical Curve("South") = {AB, BC};
Physical Curve("East") = {FA};
Physical Curve("North") = {DE, EF};
Physical Curve("West") = {CD};
Physical Surface("Domain") = {left, right};

// Mesh settings
Transfinite Curve "*" = 4 Using Progression 1;
Transfinite Surface {ABEF};
Recombine Surface {ABEF};
