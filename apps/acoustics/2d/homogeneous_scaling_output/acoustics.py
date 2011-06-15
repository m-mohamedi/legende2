#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from petsc4py import PETSc

def acoustics2D(iplot=False,htmlplot=False,use_petsc=False,outdir='./_output',solver_type='classic'):
    """
    Example python script for solving the 2d acoustics equations.
    """
    use_petsc = True

    if use_petsc:
        import petclaw as pyclaw
    else:
        import pyclaw

    if solver_type=='classic':
        solver=pyclaw.ClawSolver2D()
    elif solver_type=='sharpclaw':
        solver=pyclaw.SharpClawSolver2D()

    size = PETSc.Comm.getSize(PETSc.COMM_WORLD)
    rank = PETSc.Comm.getRank(PETSc.COMM_WORLD)
            
    
    solver.dim_split=True
    solver.mwaves = 2
    solver.limiters = [4]*solver.mwaves

    solver.mthbc_lower[0]=pyclaw.BC.reflecting
    solver.mthbc_upper[0]=pyclaw.BC.outflow
    solver.mthbc_lower[1]=pyclaw.BC.reflecting
    solver.mthbc_upper[1]=pyclaw.BC.outflow

    solver.cfl_max = 0.5
    solver.cfl_desired = 0.45
    solver.dt_variable = True    

    # Initialize grid
    mx=4*int(np.sqrt(size*10000)); my=mx
    if rank == 0:
        print "mx, my = ",mx, my    
    x = pyclaw.Dimension('x',-1.0,1.0,mx)
    y = pyclaw.Dimension('y',-1.0,1.0,my)
    grid = pyclaw.Grid([x,y])
    state = pyclaw.State(grid)

    rho = 1.0
    bulk = 4.0
    cc = np.sqrt(bulk/rho)
    zz = rho*cc
    state.aux_global['rho']= rho
    state.aux_global['bulk']=bulk
    state.aux_global['zz']= zz
    state.aux_global['cc']=cc

    state.meqn = 3

    Y,X = np.meshgrid(grid.y.center,grid.x.center)
    r = np.sqrt(X**2 + Y**2)
    width=0.2
    state.q[0,:,:] = (np.abs(r-0.5)<=width)*(1.+np.cos(np.pi*(r-0.5)/width))
    state.q[1,:,:] = 0.
    state.q[2,:,:] = 0.

    ##sol = {"n":pyclaw.Solution(state)}
    solver.dt=np.min(grid.d)/state.aux_global['cc']*solver.cfl_desired
    ##solver.setup(sol)

    # Solve
    tfinal =  0.2/np.sqrt(size)
    import time
    start=time.time()
    ##solver.evolve_to_time(sol,tfinal)
    #end=time.time()
    #duration1 = end-start
    #print 'evolve_to_time took'+str(duration1)+' seconds, for process '+str(rank)
    #if rank ==0:
        #print 'number of steps: '+ str(solver.status.get('numsteps'))
                                        


    # Solve
    claw = pyclaw.Controller()
    claw.keep_copy = False
    claw.solution = pyclaw.Solution(state)
    claw.solver = solver
    claw.outdir='./_output_'+str(size)
    claw.nout = 1
                    
    claw.tfinal =  0.2/np.sqrt(size)
    status = claw.run()

    end=time.time()
    duration1 = end-start
    print 'evolve_to_time took'+str(duration1)+' seconds, for process '+str(rank)
    if rank ==0:
        print 'number of steps: '+ str(claw.solver.status.get('numsteps'))
                        


    pressure=0
    return pressure


if __name__=="__main__":

    import sys
    from petsc4py import PETSc
    WithArgs = False
    generateProfile = True
    proccessesList = [0,5]
                        
    if len(sys.argv)>1:
        from pyclaw.util import _info_from_argv
        args, kwargs = _info_from_argv(sys.argv)
        WithArgs= True
    if generateProfile:
        rank =PETSc.Comm.getRank(PETSc.COMM_WORLD)
        size =PETSc.Comm.getSize(PETSc.COMM_WORLD)
        if rank in proccessesList:
            import cProfile
            if WithArgs: cProfile.run('acoustics2D(*args,**kwargs)', 'profile'+str(rank)+'_'+str(size))
            else: cProfile.run('acoustics2D()', 'profile'+str(rank)+'_'+str(size))
        else:
            print "process"+str(rank) +"not profiled"
            if WithArgs: acoustics2D(*args,**kwargs)
            else: acoustics2D()
    else:
        if WithArgs: acoustics2D(*args,**kwargs)
        else: acoustics2D()
