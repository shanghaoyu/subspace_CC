XLF = ftn -mp -O3

all_objects = ccm_modules.o chiral_module_andreas_with_delta.o ccm_library.o ccm_main.o ccm_iter.o ccm_energy.o ccm_t2_eqn.o ccm_diis.o ccm_mapping.o ccm_t3_channels.o ccm_triples.o ccm_t3_eqn.o ccm_t3full_channels.o ccm_v3nf_channels.o

prog_ccm.exe : ${all_objects}
	${XLF} -o prog_ccm.exe ${all_objects} 

%.o: %.f90
	${XLF} -c $<

%.o: %.f
	${XLF} -c $<

clean:
	rm *.mod *.o

