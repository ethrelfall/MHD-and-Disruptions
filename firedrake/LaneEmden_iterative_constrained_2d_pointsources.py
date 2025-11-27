# LaneEmden_iterative_constrained_2d_pointsources.py
# Ed Threlfall, November 2025
# this solves the 2D Lane-Emden equation as a simple free-boundary problem
# this only works for first-order currently

from firedrake import *
from scipy import special  # this is just so can plot analytic solution (Bessel function)

res_2d = 81
order_2d = 1
L_2d = 6.0
scale_2d = 0.01

mesh_2d = RectangleMesh(res_2d, res_2d, L_2d, L_2d, -L_2d, -L_2d, quadrilateral=False)
x_2d, y_2d = SpatialCoordinate(mesh_2d)

V_2d_s = FunctionSpace(mesh_2d, "CG", order_2d)
V_2d_c = FunctionSpace(mesh_2d, "R", 0)  # for Lagrange multiplier
V_2d = V_2d_s * V_2d_c

uj_2d = Function(V_2d)
u_aux_2d, j_2d = split(uj_2d)
(v_2d, k_2d) = split(TestFunction(V_2d))
u_2d = u_aux_2d + j_2d

# this is a constraint on the amount of matter in psi (i.e. not counting the external sources)
matter_constraint = 7.8433

u_0_2d = Constant(matter_constraint/144.0)
print("2D constraint: "+str(float(u_0_2d)))
rhs = Function(V_2d.sub(0))
bcf = Function(V_2d.sub(0))

uj_2d.sub(0).rename("solution")
rhs.rename("rhs")

# initial guess
uj_2d.sub(0).interpolate(conditional(lt(sqrt(x_2d**2+y_2d**2),pi), sin(sqrt(x_2d**2+y_2d**2))/sqrt(x_2d**2+y_2d**2), pi/sqrt(x_2d**2+y_2d**2)-1.0))  # initial guess!

# initial guess RHS matter term (derived from above)
rhs.interpolate(uj_2d.sub(0)*0.5*(1+tanh(uj_2d.sub(0)/scale_2d)))

# initial guess boundary condition
bcf.interpolate(pi/sqrt(x_2d**2+y_2d**2)-1.0)

a_2d = -dot(grad(v_2d), grad(u_2d))*dx + inner((0.5*(1+tanh((u_2d)/scale_2d)))*u_2d - u_0_2d, k_2d)*dx
l_2d = rhs*v_2d*dx
bc_2d = DirichletBC(V_2d.sub(0), bcf, "on_boundary")

u_2d_analytic = Function(V_2d_s)
# there is no analytic or MMS solution yet, add one if desired ...
# done like this as interpolate does not seem to like the special function ...
#for i0 in range(0, (res_2d+1)*(res_2d+1)):  # all points
#   x0 = mesh_2d.coordinates.dat.data[i0][0]
#   y0 = mesh_2d.coordinates.dat.data[i0][1]
#   r0 = sqrt(x0**2+y0**2)
#   u_2d_analytic.dat.data[i0] = ...

u_2d_analytic.rename("analytic solution")

# LOOP

outfile = File("LaneEmden_iterative_constrained_2d_pointsources.pvd")

for i in range(0,20):  # 10 to 20 iters seems sufficient in cases tested

   # solve
   integral_check_before = assemble((-0.5*(1+tanh((uj_2d.sub(0)+j_2d)/scale_2d)))*(uj_2d.sub(0)+j_2d)*dx)
   print("total mass before: "+str(integral_check_before))
   solve(a_2d+l_2d==0, uj_2d, bcs=bc_2d)  # be aware bc applied to u_aux not u_2d i.e. there is an offset
   print("found intermediate solution")
   integral_check_after = assemble((-0.5*(1+tanh((uj_2d.sub(0)+j_2d)/scale_2d)))*(uj_2d.sub(0)+j_2d)*dx)
   print("total mass after: "+str(integral_check_after))

   jval = assemble(j_2d*dx)/144.0

   print("j_2d: "+str(jval))

   # compute new RHS

   rhs.interpolate((uj_2d.sub(0)+jval)*0.5*(1+tanh((uj_2d.sub(0)+jval)/scale_2d)))

   eps2 = 1.0e-8

#  additional sources, pointlike (narrow Gaussian)
   ampp = 4.0
   invw2 = 4.0
   val1 = 0.0
   val2 = 4.0
   val3 = sqrt(8.0)
   rhs.interpolate(rhs+ampp*exp(-invw2*((x_2d-val1)**2+(y_2d-val2)**2)))
   rhs.interpolate(rhs-ampp*exp(-invw2*((x_2d-val3)**2+(y_2d-val3)**2)))
   rhs.interpolate(rhs+ampp*exp(-invw2*((x_2d-val2)**2+(y_2d-val1)**2)))
   rhs.interpolate(rhs-ampp*exp(-invw2*((x_2d-val3)**2+(y_2d+val3)**2)))
   rhs.interpolate(rhs+ampp*exp(-invw2*((x_2d-val1)**2+(y_2d+val2)**2)))
   rhs.interpolate(rhs-ampp*exp(-invw2*((x_2d+val3)**2+(y_2d+val3)**2)))
   rhs.interpolate(rhs+ampp*exp(-invw2*((x_2d+val2)**2+(y_2d+val1)**2)))
   rhs.interpolate(rhs-ampp*exp(-invw2*((x_2d+val3)**2+(y_2d-val3)**2)))

   # compute new boundary condition using GF

   Greens_func_2d = Function(V_2d_s)
   for i0 in range(0, (1*res_2d+1)*(1*res_2d+1)):  # all points (really should be only boundary points) - note higher orders don't work like this
      x0 = mesh_2d.coordinates.dat.data[i0][0]
      y0 = mesh_2d.coordinates.dat.data[i0][1]
      if ((abs(x0)>5.99) or (abs(y0)>5.99)):  # restrict to bdy points
         Greens_func_2d.sub(0).interpolate((1.0/(4*pi))*ln((x_2d-x0)**2+(y_2d-y0)**2+eps2))

         # convolution integrate over whole domain
         #integral = assemble(Greens_func_2d.sub(0)*((-0.5*(1+(tanh((uj_2d.sub(0)+jval)/scale_2d))))*(uj_2d.sub(0)+jval)-(ampp*exp(-((x_2d-xp)**2+(y_2d-yp)**2)))+(ampp*exp(-((x_2d-xp)**2+(y_2d-yp-0.25)**2)))+(ampp*exp(-((x_2d-xp)**2+(y_2d+yp)**2)))-(ampp*exp(-((x_2d-xp)**2+(y_2d+yp-0.25)**2))))*dx)
      
         integral = assemble(Greens_func_2d.sub(0)*((-0.5*(1+(tanh((uj_2d.sub(0)+jval)/scale_2d))))*(uj_2d.sub(0)+jval)-(ampp*exp(-invw2*((x_2d-val1)**2+(y_2d-val2)**2)))+(ampp*exp(-invw2*((x_2d-val3)**2+(y_2d-val3)**2)))-(ampp*exp(-invw2*((x_2d-val2)**2+(y_2d-val1)**2)))+(ampp*exp(-invw2*((x_2d-val3)**2+(y_2d+val3)**2)))-(ampp*exp(-invw2*((x_2d-val1)**2+(y_2d+val2)**2)))+(ampp*exp(-invw2*((x_2d+val3)**2+(y_2d+val3)**2)))-(ampp*exp(-invw2*((x_2d+val2)**2+(y_2d+val1)**2)))+(ampp*exp(-invw2*((x_2d+val3)**2+(y_2d-val3)**2))))*dx)
         #integral = assemble(Greens_func_2d.sub(0)*((-0.5*(1+(tanh((uj_2d.sub(0)+jval)/scale_2d))))*(uj_2d.sub(0)+jval)-(ampp*exp(-invw2*((x_2d-val1)**2+(y_2d-val2)**2)))+(ampp*exp(-invw2*((x_2d-val3)**2+(y_2d-val3)**2)))-(ampp*exp(-invw2*((x_2d-val2)**2+(y_2d-val1)**2)))+(ampp*exp(-invw2*((x_2d-val3)**2+(y_2d+val3)**2)))-(ampp*exp(-invw2*((x_2d-val1)**2+(y_2d+val2)**2)))+(ampp*exp(-invw2*((x_2d+val3)**2+(y_2d+val3)**2)))-(ampp*exp(-invw2*((x_2d+val2)**2+(y_2d+val1)**2)))+(ampp*exp(-invw2*((x_2d+val3)**2+(y_2d-val3)**2))))*dx)
         bcf.dat.data[i0] = integral

   # output solution
   bcf.rename("bcf")
   solution_no_offset = Function(V_2d.sub(0))
   solution_no_offset.interpolate(uj_2d.sub(0)+jval)
   solution_no_offset.rename("solution_no_offset")
   outfile.write(solution_no_offset, rhs, u_2d_analytic)
