lc = 2.;
A = newp; Point(A) = {0,0,0,lc};
B = newp; Point(B) = {1,0,0,lc};
C = newp; Point(C) = {0,1,0,lc};
O = newp; Point(O) = {0,0,1,lc};

AB = newl; Line(AB) = {A,B};
BC = newl; Line(BC) = {B,C};
CA = newl; Line(CA) = {C,A};
OA = newl; Line(OA) = {O,A};
OB = newl; Line(OB) = {O,B};
OC = newl; Line(OC) = {O,C};

ll_ABC = newll; Line Loop(ll_ABC) = {AB, BC, CA};
ll_ABO = newll; Line Loop(ll_ABO) = {AB, -OB, OA};
ll_ACO = newll; Line Loop(ll_ACO) = {CA, -OA, OC};
ll_BCO = newll; Line Loop(ll_BCO) = {BC, -OC, OB};

ABC = news; Plane Surface(ABC) = {ll_ABC};
ABO = news; Plane Surface(ABO) = {ll_ABO};
ACO = news; Plane Surface(ACO) = {ll_ACO};
BCO = news; Plane Surface(BCO) = {ll_BCO};

sl = newsl; Surface Loop(sl) = {ABC, ABO, ACO, BCO};
vol = newv; Volume(vol) = {sl};


