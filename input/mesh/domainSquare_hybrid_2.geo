//    M11---------M12--------------------------M13-------M14
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    M21---------M22--------------------------M23-------M24
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    M31---------M32--------------------------M33-------M34
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    |           |                            |         |
//    M41---------M42--------------------------M43-------M44

// Geometrical parameters
xmin = 0.;
xmax = 1.;
ymin = 0.;
ymax = 1.;
lx_1_2 = 0.2; // in percent of (xmax - xmin)
lx_3_4 = 0.2; // in percent of (xmax - xmin)
ly_1_2 = 0.2; // in percent of (ymax - ymin)
ly_3_4 = 0.2; // in percent of (ymax - ymin)

// Mesh settings
lc = 0.05;
nx_1 = 8;
nx_2 = 12; // main section
nx_3 = 8;
ny_1 = 8;
ny_2 = 12; // main section
ny_3 = 8;

//--------------------------------------------------------------- Length
lx = xmax - xmin;
ly = ymax - ymin;

//--------------------------------------------------------------- Points

//-- Row 1
M11 = newp; Point(M11) = {xmin              , ymax, 0., lc};
M12 = newp; Point(M12) = {xmin + lx_1_2 * lx, ymax, 0., lc};
M13 = newp; Point(M13) = {xmax - lx_3_4 * lx, ymax, 0., lc};
M14 = newp; Point(M14) = {xmax              , ymax, 0., lc};

//-- Row 2
M21 = newp; Point(M21) = {xmin              , ymax - ly_1_2 * ly, 0., lc};
M22 = newp; Point(M22) = {xmin + lx_1_2 * lx, ymax - ly_1_2 * ly, 0., lc};
M23 = newp; Point(M23) = {xmax - lx_3_4 * lx, ymax - ly_1_2 * ly, 0., lc};
M24 = newp; Point(M24) = {xmax              , ymax - ly_1_2 * ly, 0., lc};

//-- Row 3
M31 = newp; Point(M31) = {xmin              , ymin + ly_3_4 * ly, 0., lc};
M32 = newp; Point(M32) = {xmin + lx_1_2 * lx, ymin + ly_3_4 * ly, 0., lc};
M33 = newp; Point(M33) = {xmax - lx_3_4 * lx, ymin + ly_3_4 * ly, 0., lc};
M34 = newp; Point(M34) = {xmax              , ymin + ly_3_4 * ly, 0., lc};

//-- Row 4
M41 = newp; Point(M41) = {xmin              , ymin, 0., lc};
M42 = newp; Point(M42) = {xmin + lx_1_2 * lx, ymin, 0., lc};
M43 = newp; Point(M43) = {xmax - lx_3_4 * lx, ymin, 0., lc};
M44 = newp; Point(M44) = {xmax              , ymin, 0., lc};

//---------------------------------------------------------------- Lines
//-- Horizontal lines
L_11_12 = newl; Line(L_11_12) = {M11, M12};
L_12_13 = newl; Line(L_12_13) = {M12, M13};
L_13_14 = newl; Line(L_13_14) = {M13, M14};
L_21_22 = newl; Line(L_21_22) = {M21, M22};
L_22_23 = newl; Line(L_22_23) = {M22, M23};
L_23_24 = newl; Line(L_23_24) = {M23, M24};
L_31_32 = newl; Line(L_31_32) = {M31, M32};
L_32_33 = newl; Line(L_32_33) = {M32, M33};
L_33_34 = newl; Line(L_33_34) = {M33, M34};
L_41_42 = newl; Line(L_41_42) = {M41, M42};
L_42_43 = newl; Line(L_42_43) = {M42, M43};
L_43_44 = newl; Line(L_43_44) = {M43, M44};

//-- Vertical lines
L_11_21 = newl; Line(L_11_21) = {M11, M21};
L_21_31 = newl; Line(L_21_31) = {M21, M31};
L_31_41 = newl; Line(L_31_41) = {M31, M41};
L_12_22 = newl; Line(L_12_22) = {M12, M22};
L_22_32 = newl; Line(L_22_32) = {M22, M32};
L_32_42 = newl; Line(L_32_42) = {M32, M42};
L_13_23 = newl; Line(L_13_23) = {M13, M23};
L_23_33 = newl; Line(L_23_33) = {M23, M33};
L_33_43 = newl; Line(L_33_43) = {M33, M43};
L_14_24 = newl; Line(L_14_24) = {M14, M24};
L_24_34 = newl; Line(L_24_34) = {M24, M34};
L_34_44 = newl; Line(L_34_44) = {M34, M44};

//------------------------------------------------------------ Line loop
LL_11 = newll; Line Loop(LL_11) = {L_21_22, - L_12_22, - L_11_12, L_11_21};
LL_12 = newll; Line Loop(LL_12) = {L_22_23, - L_13_23, - L_12_13, L_12_22};
LL_13 = newll; Line Loop(LL_13) = {L_23_24, - L_14_24, - L_13_14, L_13_23};

LL_21 = newll; Line Loop(LL_21) = {L_31_32, - L_22_32, - L_21_22, L_21_31};
LL_22 = newll; Line Loop(LL_22) = {L_32_33, - L_23_33, - L_22_23, L_22_32};
LL_23 = newll; Line Loop(LL_23) = {L_33_34, - L_24_34, - L_23_24, L_23_33};

LL_31 = newll; Line Loop(LL_31) = {L_41_42, - L_32_42, - L_31_32, L_31_41};
LL_32 = newll; Line Loop(LL_32) = {L_42_43, - L_33_43, - L_32_33, L_32_42};
LL_33 = newll; Line Loop(LL_33) = {L_43_44, - L_34_44, - L_33_34, L_33_43};


//-------------------------------------------------------------- Surface
S_11 = news; Plane Surface(S_11) = {LL_11};
S_12 = news; Plane Surface(S_12) = {LL_12};
S_13 = news; Plane Surface(S_13) = {LL_13};

S_21 = news; Plane Surface(S_21) = {LL_21};
S_22 = news; Plane Surface(S_22) = {LL_22};
S_23 = news; Plane Surface(S_23) = {LL_23};

S_31 = news; Plane Surface(S_31) = {LL_31};
S_32 = news; Plane Surface(S_32) = {LL_32};
S_33 = news; Plane Surface(S_33) = {LL_33};


//----------------------------------------------------------------- Mesh
Transfinite Line {L_11_12, L_21_22, L_31_32, L_41_42} = nx_1;
Transfinite Line {L_12_13, L_22_23, L_32_33, L_42_43} = nx_2;
Transfinite Line {L_13_14, L_23_24, L_33_34, L_43_44} = nx_3;

Transfinite Line {L_11_21, L_12_22, L_13_23, L_14_24} = ny_1;
Transfinite Line {L_21_31, L_22_32, L_23_33, L_24_34} = ny_2;
Transfinite Line {L_31_41, L_32_42, L_33_43, L_34_44} = ny_3;

Transfinite Surface {S_11, S_12, S_13, S_21, S_23, S_31, S_32, S_33};
Recombine Surface {S_11, S_12, S_13, S_21, S_23, S_31, S_32, S_33};

//--------------------------------------------------------- Physical tag
Physical Point("xmin_ymin") = {M41};
Physical Point("xmin_ymax") = {M11};
Physical Point("xmax_ymin") = {M44};
Physical Point("xmax_ymax") = {M14};
Physical Line("xmin") = {L_11_21, L_21_31, L_31_41};
Physical Line("xmax") = {L_14_24, L_24_34, L_34_44};
Physical Line("ymin") = {L_41_42, L_42_43, L_43_44};
Physical Line("ymax") = {L_11_12, L_12_13, L_13_14};
Physical Surface("INTERIOR") = {S_11, S_12, S_13, S_21, S_22, S_23, S_31, S_32, S_33};
