lc = 2.;
A = newp; Point(A) = {0,0,-1,lc};
B = newp; Point(B) = {1,0,-1,lc};
C = newp; Point(C) = {0,1,-1,lc};

AB = newl; Line(AB) = {A,B};
BC = newl; Line(BC) = {B,C};
CA = newl; Line(CA) = {C,A};

ll_ABC = newll; Line Loop(ll_ABC) = {AB, BC, CA};

ABC = news; Plane Surface(ABC) = {ll_ABC};

out[] = Extrude {0,0,2}{Surface{ABC}; Layers{1}; Recombine;};