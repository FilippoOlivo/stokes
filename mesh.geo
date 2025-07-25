L = 5.0;
H = 1.5;
R = 0.1;
cx = 0.5;
cy = 1;

lc_wall = 0.075;
lc_cylinder = 0.0375;

// Outer rectangle
Point(1) = {0, 0, 0, lc_wall};
Point(2) = {L, 0, 0, lc_wall};
Point(3) = {L, H, 0, lc_wall};
Point(4) = {0, H, 0, lc_wall};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(10) = {1, 2, 3, 4};

// Cylinder 1
Point(100) = {cx, cy, 0, lc_cylinder};
Point(101) = {cx+R, cy, 0, lc_cylinder};
Point(102) = {cx, cy+R, 0, lc_cylinder};
Point(103) = {cx-R, cy, 0, lc_cylinder};
Point(104) = {cx, cy-R, 0, lc_cylinder};

Circle(201) = {101, 100, 102};
Circle(202) = {102, 100, 103};
Circle(203) = {103, 100, 104};
Circle(204) = {104, 100, 101};
Line Loop(20) = {201, 202, 203, 204};


cx = 1.5;
cy = 0.5;

// Cylinder 2
Point(1100) = {cx, cy, 0, lc_cylinder};
Point(1101) = {cx+R, cy, 0, lc_cylinder};
Point(1102) = {cx, cy+R, 0, lc_cylinder};
Point(1103) = {cx-R, cy, 0, lc_cylinder};
Point(1104) = {cx, cy-R, 0, lc_cylinder};

Circle(1201) = {1101, 1100, 1102};
Circle(1202) = {1102, 1100, 1103};
Circle(1203) = {1103, 1100, 1104};
Circle(1204) = {1104, 1100, 1101};
Line Loop(120) = {1201, 1202, 1203, 1204};


// Surface with hole
Plane Surface(30) = {10, 20, 120};
Recombine Surface{30};

// Physical entities
Physical Line(10) = {4};                       // Inlet
Physical Line(20) = {2};                       // Outlet
Physical Line(30) = {1, 3};                    // Walls
Physical Line(40) = {201, 202, 203, 204, 1201, 1202, 1203, 1204};      // Cylinder
Physical Surface(1) = {30};                    // Fluid domain
