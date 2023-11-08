//        G------H
//        |      |
//        |      |
// D------E------F
// |      |      |
// |      |      |
// A------B------C
//
// ABED and EFHG are extruded with on layer, BCFE is extruded with two layers

lc = 1.;

// Points
A = newp; Point(A) = {0,0,0,lc};
B = newp; Point(B) = {1,0,0,lc};
C = newp; Point(C) = {2,0,0,lc};
D = newp; Point(D) = {0,1,0,lc};
E = newp; Point(E) = {1,1,0,lc};
F = newp; Point(F) = {2,1,0,lc};
G = newp; Point(G) = {1,2,0,lc};
H = newp; Point(H) = {2,2,0,lc};

// Lines
AB = newl; Line(AB) = {A,B};
BC = newl; Line(BC) = {B,C};
DE = newl; Line(DE) = {D,E};
EF = newl; Line(EF) = {E,F};
GH = newl; Line(GH) = {G,H};
AD = newl; Line(AD) = {A,D};
BE = newl; Line(BE) = {B,E};
CF = newl; Line(CF) = {C,F};
EG = newl; Line(EG) = {E,G};
FH = newl; Line(FH) = {F,H};

// Line loops
_ABED = newll; Line Loop(_ABED) = {AB, BE, -DE, -AD};
_BCFE = newll; Line Loop(_BCFE) = {BC, CF, -EF, -BE};
_EFHG = newll; Line Loop(_EFHG) = {EF, FH, -GH, -EG};

// Surfaces
ABED = news; Plane Surface(ABED) = {_ABED};
BCFE = news; Plane Surface(BCFE) = {_BCFE};
EFHG = news; Plane Surface(EFHG) = {_EFHG};

// Extrusion
out_1[] = Extrude {0,0,1}{Surface{ABED, BCFE, EFHG}; Layers{1}; Recombine;};
// out_1[0] est le top de ABED extrude
// out_1[1] est le volume cree par l'extrusion de ABED
// out_1[2] est le side de l'extrusion de AB (=premier element de ABED)
// out_1[3] est le side de l'extrusion de BE (=deuxieme element de ABED)
// out_1[4] est le side de l'extrusion de ED (=troisieme element de ABED)
// out_1[5] est le side de l'extrusion de DA (=quatrieme element de ABED)
// out_1[6] est le top de BCFE extrude
// out_1[7] est le volume cree par l'extrusion de BCFE
// out_1[8] est le side de l'extrusion de BC (=premier element de BCFE)
// out_1[9] est le side de l'extrusion de CF (=deuxieme element de BCFE)
// out_1[10] est le side de l'extrusion de FE (=troisieme element de BCFE)
// out_1[11] est le side de l'extrusion de EB (=quatrieme element de BCFE)
// out_1[12] est le top de EFHG extrude
// out_1[13] est le volume cree par l'extrusion de EFHG
// out_1[14] est le side de l'extrusion de EF (=premier element de EFHG)
// out_1[15] est le side de l'extrusion de FH (=deuxieme element de EFHG)
// out_1[16] est le side de l'extrusion de HG (=troisieme element de EFHG)
// out_1[17] est le side de l'extrusion de GE (=quatrieme element de EFHG)

out_2[] = Extrude {0,0,1}{Surface{out_1[6]}; Layers{1}; Recombine;};

// Mesh settings
Transfinite Line{AB, BC, DE, EF, GH, AD, BE, CF, EG, FH} = 2;
Transfinite Surface{ABED, BCFE, EFHG};
Recombine Surface{ABED, BCFE, EFHG};