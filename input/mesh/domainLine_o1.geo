width = 2.;
A = newp; Point(A) = {-width/2., 0., 0.};
B = newp; Point(B) = { width/2., 0., 0.};

AB = newl; Line(AB) = {A, B};

Transfinite Line{AB} = 11;

Mesh.ElementOrder = 1;
