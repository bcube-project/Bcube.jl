// D------E------F
// |      |      |
// |      |      |
// A------B------C

lc = 1.;

// Points
A = newp; Point(A) = {0,0,0,lc};
B = newp; Point(B) = {1,0,0,lc};
C = newp; Point(C) = {2,0,0,lc};
D = newp; Point(D) = {0,1,0,lc};
E = newp; Point(E) = {1,1,0,lc};
F = newp; Point(F) = {2,1,0,lc};

// Lines
AB = newl; Line(AB) = {A,B};
BC = newl; Line(BC) = {B,C};
DE = newl; Line(DE) = {D,E};
EF = newl; Line(EF) = {E,F};
AD = newl; Line(AD) = {A,D};
BE = newl; Line(BE) = {B,E};
CF = newl; Line(CF) = {C,F};

// Line loops
_ABED = newll; Line Loop(_ABED) = {AB, BE, -DE, -AD};
_BCFE = newll; Line Loop(_BCFE) = {BC, CF, -EF, -BE};

// Surfaces
ABED = news; Plane Surface(ABED) = {_ABED};
BCFE = news; Plane Surface(BCFE) = {_BCFE};

// Extrusion
out[] = Extrude {0,0,1}{Surface{ABED, BCFE}; Layers{1}; Recombine;};
// out[0] est le top de ABED extrude
// out[1] est le volume cree par l'extrusion de ABED
// out[2] est le side de l'extrusion de AB (=premier element de ABED)
// out[3] est le side de l'extrusion de BE (=deuxieme element de ABED)
// out[4] est le side de l'extrusion de ED (=troisieme element de ABED)
// out[5] est le side de l'extrusion de DA (=quatrieme element de ABED)
// out[6] est le top de BCFE extrude
// out[7] est le volume cree par l'extrusion de BCFE
// out[8] est le side de l'extrusion de BC (=premier element de BCFE)
// out[9] est le side de l'extrusion de CF (=deuxieme element de BCFE)
// out[10] est le side de l'extrusion de FE (=troisieme element de BCFE)
// out[11] est le side de l'extrusion de EB (=quatrieme element de BCFE)

// Mesh settings
Transfinite Line{AB, BC, DE, EF, AD, BE, CF} = 2;
Transfinite Surface{ABED, BCFE};
Recombine Surface{ABED, BCFE};