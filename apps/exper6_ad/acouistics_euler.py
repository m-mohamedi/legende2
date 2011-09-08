
def acoiustics_main():
    from time import time
    tstart = time()
    import sys
    from petsc4py import PETSc
    import cProfile
                        
    WithArgs = False
    generateProfile = True
    proccessesList = [0,5]

    rank =PETSc.Comm.getRank(PETSc.COMM_WORLD)
    size =PETSc.Comm.getSize(PETSc.COMM_WORLD)
    tb1call=time()        
    acoustics2D(finalt=(0.2/np.sqrt(size))/40,use_petsc=True,outdir='exper6_a/_output_p1')
    PETSc.COMM_WORLD.barrier()
    
    tb2call=time()
    rank =PETSc.Comm.getRank(PETSc.COMM_WORLD)
    size =PETSc.Comm.getSize(PETSc.COMM_WORLD)    
    acoustics2D(finalt=(0.2/np.sqrt(size))/40,use_petsc=True,outdir='exper6_a/_output_p2')
    PETSc.COMM_WORLD.barrier()
    tb3call=time()
    if rank in proccessesList:
        funccall = "acoustics2D(finalt=0.2/np.sqrt("+str(size)+"),use_petsc=True,outdir='exper6_a/_output_p3')"
        cProfile.run(funccall, 'exper6_a/profile'+str(rank)+'_'+str(size))
    else:
        print "process"+str(rank) +"not profiled"
        acoustics2D(finalt=0.2/np.sqrt(size),use_petsc=True,outdir='exper6_a/_output_p3')
    PETSc.COMM_WORLD.barrier()
    tend = time()
    if rank == 0: 
        print "total time for proc",rank," is ",tend-tstart, "up to before p1",tb1call-tstart, "p1",tb2call-tb1call , "p2", tb3call-tb2call, "p3", tend-tb3call
        print "time to subtract from job time to give load time= part2*2+the rest of the code time", 2*(tb3call-tb2call)+ (tend- tb3call)

def euler_main():

    from time import time
    tstart = time()
    import sys
    from petsc4py import PETSc
    import cProfile

    WithArgs = False
    generateProfile = True
    proccessesList = [0,5]

    rank =PETSc.Comm.getRank(PETSc.COMM_WORLD)
    size =PETSc.Comm.getSize(PETSc.COMM_WORLD)
    tb1call=time()
    shockbubble(finalt=(0.02/np.sqrt(size))/40,use_petsc=True,outdir='exper6_d/_output_p1')
    PETSc.COMM_WORLD.barrier()

    tb2call=time()
    rank =PETSc.Comm.getRank(PETSc.COMM_WORLD)
    size =PETSc.Comm.getSize(PETSc.COMM_WORLD)
    shockbubble(finalt=(0.02/np.sqrt(size))/40,use_petsc=True,outdir='exper6_d/_output_p2')
    PETSc.COMM_WORLD.barrier()
    tb3call=time()
    if rank in proccessesList:
        funccall = "shockbubble(finalt=0.02/np.sqrt("+str(size)+"),use_petsc=True,outdir='exper6_d/_output_p3')"
        cProfile.run(funccall, 'exper6_d/profile'+str(rank)+'_'+str(size))
    
    else:
        print "process"+str(rank) +"not profiled"
        shockbubble(finalt=0.02/np.sqrt(size),use_petsc=True,outdir='exper6_d/_output_p3')
    
    PETSc.COMM_WORLD.barrier()
    tend = time()
    if rank == 0:
        print "total time for proc",rank," is ",tend-tstart, "up to before p1",tb1call-tstart, "p1",tb2call-tb1call , "p2", tb3call-tb2call, "p3", tend-tb3call
        print "time to subtract from job time to give load time= part2*2+the rest of the code time", 2*(tb3call-tb2call)+ (tend- tb3call)
                                                                                                                                                      
if __name__ == "__main__":

    import numpy as np
    from petsc4py import PETSc
    import sys
    sys.path.append("../acoustics/2d/exper6_weak") #acouistics classic2.so path
    #import classic2
    #print "dir(classic2), dir(classic2.cparam), dir(classic2.comxyt)", dir(classic2), dir(classic2.cparam), dir(classic2.comxyt)
    # run acouistics example
    from acoustics import acoustics2D
    acoiustics_main()
    #run ../acoustics/2d/exper6_weak/acoustics.py
    sys.path.remove("../acoustics/2d/exper6_weak") # acouistics classic2.so path
    sys.path.append("../euler/exper6_d_weak") # euler classic2.so path
    #reload(sys.modules["classic2"])
    #print "dir(classic2), dir(classic2.cparam), dir(classic2.param), dir(classic2.comxyt),dir(classic2.comroe)", dir(classic2), dir(classic2.cparam), dir(classic2.param), dir(classic2.comxyt),dir(classic2.comroe)
    # run euler example
    #run ../euler/exper6_d_weak/shockbubble.py
    from shockbubble import shockbubble
    euler_main()

