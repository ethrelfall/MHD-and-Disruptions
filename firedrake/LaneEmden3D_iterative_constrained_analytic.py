# LaneEmden3D_iterative_constrained_analytic.py
# Ed Threlfall, November 2025
# this solves the 3D Lane-Emden equation as a simple free-boundary problem (sin(r)/r then pi/r-1 for r>pi)
# this only works for first-order currently
# this is set up to run at low res; construction of the Green's function does not scale well

from firedrake import *

res_3d = 11
order_3d = 1
L_3d = 6.0  # this is half side length
scale_3d = 0.01

mesh_3d = BoxMesh(res_3d, res_3d, res_3d, 2*L_3d, 2*L_3d, 2*L_3d)
x_3d, y_3d, z_3d = SpatialCoordinate(mesh_3d)

V_3d_s = FunctionSpace(mesh_3d, "CG", order_3d)
V_3d_c = FunctionSpace(mesh_3d, "R", 0)  # for Lagrange multiplier
V_3d = V_3d_s * V_3d_c

uj_3d = Function(V_3d)
u_aux_3d, j_3d = split(uj_3d)
(v_3d, k_3d) = split(TestFunction(V_3d))
u_3d = u_aux_3d + j_3d

# this is a constraint on the amount of matter in psi (i.e. not counting the external sources)
matter_constraint = 4*pi*pi

u_0_3d = Constant(matter_constraint/1728.0)
print("3D constraint: "+str(float(u_0_3d)))
rhs = Function(V_3d.sub(0))
bcf = Function(V_3d.sub(0))

uj_3d.sub(0).rename("solution")
rhs.rename("rhs")

eps2_init = 0.0e-8

# initial guess
uj_3d.sub(0).interpolate(1.0*conditional(lt(sqrt((x_3d-6.0)**2+(y_3d-6.0)**2+(z_3d-6.0)**2),pi), sin(sqrt((x_3d-6.0)**2+(y_3d-6.0)**2+(z_3d-6.0)**2))/sqrt((x_3d-6.0)**2+(y_3d-6.0)**2+(z_3d-6.0)**2+eps2_init), pi/sqrt((x_3d-6.0)**2+(y_3d-6.0)**2+(z_3d-6.0)**2)-1.0))  # initial guess!

# initial guess RHS matter term (derived from above)
rhs.interpolate(uj_3d.sub(0)*0.5*(1+tanh(uj_3d.sub(0)/scale_3d)))

# initial guess boundary condition
bcf.interpolate(1.0*pi/sqrt((x_3d-6.0)**2+(y_3d-6.0)**2+(z_3d-6.0)**2)-1.0)

a_3d = -dot(grad(v_3d), grad(u_3d))*dx + inner((0.5*(1+tanh((u_3d)/scale_3d)))*u_3d - u_0_3d, k_3d)*dx
l_3d = rhs*v_3d*dx
bc_3d = DirichletBC(V_3d.sub(0), bcf, "on_boundary")

u_3d_analytic = Function(V_3d_s)
u_3d_analytic.interpolate(conditional(lt(sqrt((x_3d-6.0)**2+(y_3d-6.0)**2+(z_3d-6.0)**2),pi), sin(sqrt((x_3d-6.0)**2+(y_3d-6.0)**2+(z_3d-6.0)**2))/sqrt((x_3d-6.0)**2+(y_3d-6.0)**2+(z_3d-6.0)**2+eps2_init), pi/sqrt((x_3d-6.0)**2+(y_3d-6.0)**2+(z_3d-6.0)**2)-1.0))  # initial guess!


u_3d_analytic.rename("analytic solution")

# LOOP

outfile = File("LaneEmden3D_iterative_constrained_analytic.pvd")

for i in range(0,10):


   # solve
   integral_check_before = assemble((-0.5*(1+tanh((uj_3d.sub(0)+j_3d)/scale_3d)))*(uj_3d.sub(0)+j_3d)*dx)
   print("total mass before: "+str(integral_check_before))
   solve(a_3d+l_3d==0, uj_3d, bcs=bc_3d)  # be aware bc applied to u_aux not u_2d i.e. there is an offset
   print("found intermediate solution")
   integral_check_after = assemble((-0.5*(1+tanh((uj_3d.sub(0)+j_3d)/scale_3d)))*(uj_3d.sub(0)+j_3d)*dx)
   print("total mass after: "+str(integral_check_after))
   jval = assemble(j_3d*dx)/1728.0

   print("j_3d: "+str(jval))

   # compute new RHS
   rhs.interpolate((uj_3d.sub(0)+jval)*0.5*(1+tanh((uj_3d.sub(0)+jval)/scale_3d)))


   # compute new boundary condition using GF
   eps2 = 1.0e-8

   Greens_func_3d = Function(V_3d_s)
   for i0 in range(0, (1*res_3d+1)*(1*res_3d+1)*(1*res_3d+1)):  # all points (really should be only boundary points) - note higher orders don't work like this
      x0 = mesh_3d.coordinates.dat.data[i0][0]-6.0
      y0 = mesh_3d.coordinates.dat.data[i0][1]-6.0
      z0 = mesh_3d.coordinates.dat.data[i0][2]-6.0
      if ((abs(x0)>5.9) or (abs(y0)>5.9) or (abs(z0)>5.9)):  # restrict to bdy points
         # don't forget offset needed for the x_3d, y_3d, z_3d ...
         Greens_func_3d.sub(0).interpolate((1.0/(4*pi))*1.0/sqrt((x_3d-x0-6.0)**2+(y_3d-y0-6.0)**2+(z_3d-z0-6.0)**2+eps2))

         # basic no source:
         integral = assemble((Greens_func_3d.sub(0)*((-0.5*(1+(tanh((uj_3d.sub(0)+jval)/scale_3d))))*(uj_3d.sub(0)+jval)))*dx)
         bcf.dat.data[i0] = integral  # TRIALCODE comment out so it always has the analytic bcf

   # output solution
   bcf.rename("bcf")
   solution_no_offset = Function(V_3d.sub(0))
   solution_no_offset.interpolate(uj_3d.sub(0)+jval)
   solution_no_offset.rename("solution_no_offset")
   outfile.write(solution_no_offset, rhs, u_3d_analytic, bcf)
