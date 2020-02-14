LOCAL_LIBS = -lblas -llapack 
#COMP = mpif90 -openmpi-mp -fopenmp -fdiagnostics-color=always
COMP = mpif90 -fopenmp  -ffree-line-length-1024 -fdiagnostics-color=always
FC=${COMP} -O3 
#FCD=${COMP} -Og -g -Wall -Wextra -fbounds-check -ftrapv  -Wall -pedantic-errors# -Wuse-without-only
#FCD=${COMP} -Og -g -fbounds-check  
FCD=${COMP}   

LIBS= ${LOCAL_LIBS} 

XLF = ${FCD}


all_objects =  ccm_modules_jwg.o chiral_module_andreas_with_delta.o minnesota_module.o ccm_library.o ccm_main_jwg.o ccm_iter_jwg_2.o ccm_energy.o ccm_t2_eqn.o ccm_diis.o ccm_mapping_jwg.o ccm_t3_channels.o ccm_triples_jwg.o ccm_t3_eqn.o ccm_t3full_channels_jwg.o ccm_v3nf_channels.o ccm_general_eigvalue.o IHS_sampling.o sm_benchmark.o 

prog_ccm.exe : ${all_objects}
	${XLF} -o prog_ccm.exe -L/usr/local/opt/openblas/lib  -I/usr/local/opt/openblas/include  ${all_objects} ${LIBS} 
	cp prog_ccm.exe test/prog_ccm.exe

%.o: %.f90
	${XLF} -c $<

%.o: %.f
	${XLF} -c $<

clean:
	rm *.mod *.o


