lc = 0.1;
A = newp; Point(A) = {-1,-1,-1,lc};
B = newp; Point(B) = { 1,-1,-1,lc};
C = newp; Point(C) = { 1, 1,-1,lc};
D = newp; Point(D) = {-1, 1,-1,lc};

AB = newl; Line(AB) = {A,B};
BC = newl; Line(BC) = {B,C};
CD = newl; Line(CD) = {C,D};
DA = newl; Line(DA) = {D,A};

ll = newll; Line Loop(ll) = {AB,BC,CD,DA};
ABCD = news; Plane Surface(ABCD) = {ll};

out[] = Extrude {0,0,2}{Surface{ABCD}; Layers{1}; Recombine;};

Transfinite Line {AB, BC, CD, DA} = 2;
Transfinite Surface{ABCD};
Recombine Surface{ABCD};


