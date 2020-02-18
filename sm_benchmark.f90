
!SUBROUTINE generate_sm_configuration
!  use shell_model
!
!  IMPLICIT NONE
!
!  print *,'wtf!!!', all_orbit%total_orbits
!
!
!END SUBROUTINE generate_sm_configuration


SUBROUTINE sm_calculation
  USE PARALLEL
  USE subspace
  USE single_particle_orbits
  USE one_body_operators
  USE t2_storage
  USE constants
  USE configurations
  use shell_model
  use chiral_potentials

  IMPLICIT NONE
  real*8  ::  test, cutoff
  INTEGER :: p,q,r,s,temp_config, bar, ket, INFO, loop1
  INTEGER, allocatable ::  min_ev_index(:)
  real*8,allocatable  :: WR(:),WI(:), VL(:,:), VR(:,:), WORK(:), min_ev

  ! generate all configuration 
  print *,'wtf!!!', all_orbit%total_orbits
  print *,'hh_channle', channels%number_hhhh_confs
  print *,'hhpp_channle', channels%number_hhpp_confs
  print *,'pppp_channle', channels%number_pppp_confs

  print *,'hh_hhhh_config_num',size(  lookup_hhhh_configs(1,1)%ival, 2)
  print *,'pp_pppp_config_num',size(  lookup_pppp_configs(1,1)%ival, 2)

  sm_config_num = 0

  do p = 1, all_orbit%total_orbits
     do q = p, all_orbit%total_orbits
        if (p == q)cycle
        if (all_orbit%nx(p)+all_orbit%nx(q) /= 0) cycle
        if (all_orbit%ny(p)+all_orbit%ny(q) /= 0) cycle
        if (all_orbit%nz(p)+all_orbit%nz(q) /= 0) cycle
        sm_config_num = sm_config_num + 1
     end do
  end do

  print *,'sm_config_num', sm_config_num

  if( .not. allocated(H_sm)) allocate (H_sm(sm_config_num,sm_config_num))
  if( .not. allocated(lookup_config_sm)) allocate (lookup_config_sm(2,sm_config_num))

  temp_config = 1
  do p = 1, all_orbit%total_orbits
     do q = p, all_orbit%total_orbits
        if (p == q)cycle
        if (all_orbit%nx(p)+all_orbit%nx(q) /= 0) cycle
        if (all_orbit%ny(p)+all_orbit%ny(q) /= 0) cycle
        if (all_orbit%nz(p)+all_orbit%nz(q) /= 0) cycle
        lookup_config_sm(1,temp_config) = p
        lookup_config_sm(2,temp_config) = q
        temp_config = temp_config + 1
     end do
  end do


!  do bar = 1, sm_config_num
!     print *,'bar=',bar
!     print *,lookup_config_sm(1,bar),lookup_config_sm(2,bar)
!  end do

  H_sm = 0
  do bar = 1 , sm_config_num
     do ket = 1 , sm_config_num
        p = lookup_config_sm(1,bar)
        q = lookup_config_sm(2,bar)
        r = lookup_config_sm(1,ket)
        s = lookup_config_sm(2,ket)
        !print *, p,q,r,s

        H_sm(bar,ket) =  ( chiral_NN_with_delta(p,q,r,s) - chiral_NN_with_delta(p,q,s,r) )
     end do
  end do
!  print *, H_sm
  if ( .not. allocated(WR)) allocate(WR(sm_config_num))
  if ( .not. allocated(WI)) allocate(WI(sm_config_num))
  if ( .not. allocated(VL)) allocate(VL(sm_config_num,sm_config_num))
  if ( .not. allocated(VR)) allocate(VR(sm_config_num,sm_config_num))
  if ( .not. allocated(WORK)) allocate(WORK(sm_config_num*4))

  call dgeev('N','V',sm_config_num,H_sm,sm_config_num,WR,WI,VL,sm_config_num,VR,sm_config_num,WORK,sm_config_num*4,INFO)


  if ( .not. allocated(min_ev_index)) allocate(min_ev_index(sm_config_num))
!  min_ev_index = minloc(WR,MASK = WR > -83)
! cutoff = -100000000000.D0
 ! min_ev_index = minloc(WR)
 ! print *, 'min_ev_index=', min_ev_index(1)
  cutoff = -1000000
  do loop1 = 1, sm_config_num
     min_ev_index = minloc(WR, MASK=(WR  > cutoff))
     if (  WI(min_ev_index(1)) > 0.0001 ) then
        cutoff = WR(min_ev_index(1))
     else
        exit
     end if
  end do

!  print *, 'min_ev_index=', WR(min_ev_index(1))
!  print *, 'min_evector=', VR(:,min_ev_index(1))

  if(.not. allocated(sm_evector)) allocate(sm_evector(sm_config_num))
  sm_evalue = WR(min_ev_index(1))
  sm_evector = VR(:,min_ev_index(1))

! print *, 'min_ev=', WR(min_ev_index)
!  print *, 'EValue=', WR, '+i',WI
!  print *, 'EVector=', VR
  print *, 'INFO=', INFO

  open (223,file="sm_result.txt")
  if ( iam == 0 ) write(223,*) 'sm_cal=', sm_evalue
  close(223)

END SUBROUTINE sm_calculation

SUBROUTINE print_sm_wf(subspace_count)
  USE PARALLEL
  USE CONSTANTS
  use configurations
  use deltafull_parameters
  use shell_model

  IMPLICIT NONE
  INTEGER :: bra, ket, subspace_count
  character(len=100) :: sm_file,str_temp

  !!! print sm wave fuction
  write(str_temp,"(i3)")subspace_count
  str_temp = adjustl(str_temp)
  sm_file=trim(str_temp)//'_sm.txt'

  if ( iam == 0 ) write(6,*) 'sm_file=', sm_file,"end"
  open (223,file=sm_file)

130  format (F20.15)
  do bra = 1, sm_config_num
     if ( iam == 0 )  write(223,130) sm_evector(bra)
  end do
  if ( iam == 0 ) write(223,*) 'dens=', dens
  if ( iam == 0 ) write(223,*) 'LEC=', LEC_c1_input, LEC_c2_input ,LEC_c3_input, LEC_c4_input
  if ( iam == 0 ) write(223,*) 'c1s0, c3s1 ', c1s0_input , c3s1_input
  if ( iam == 0 ) write(223,*) 'cnlo_pw(1-7) ',cnlo_pw_input
  if ( iam == 0 ) write(223,*) 'cD,cE ',cD_input, cE_input
  if ( iam == 0 ) write(223,*) 'sm_evalue=', sm_evalue
  close(223)
END SUBROUTINE print_sm_wf


SUBROUTINE setup_subspace_allocation_sm
  use shell_model
  USE single_particle_orbits

  IMPLICIT NONE
  real*8  ::  test, cutoff
  INTEGER :: p,q,r,s,temp_config, bar, ket, INFO, loop1

  
  print *,'wtf!!!', all_orbit%total_orbits

! get config number for configs with 0 momentum
  sm_config_num = 0 
  do p = 1, all_orbit%total_orbits
     do q = p, all_orbit%total_orbits
        if (p == q)cycle
        if (all_orbit%nx(p)+all_orbit%nx(q) /= 0) cycle
        if (all_orbit%ny(p)+all_orbit%ny(q) /= 0) cycle
        if (all_orbit%nz(p)+all_orbit%nz(q) /= 0) cycle
        sm_config_num = sm_config_num + 1 
     end do
  end do

  print *,'sm_config_num', sm_config_num

  if ( .not. allocated(sm_evector_subspace)) allocate(sm_evector_subspace(subspace_num_sm,sm_config_num)) 
  if ( .not. allocated(H_matrix_sm)) allocate(H_matrix_sm(subspace_num_sm,subspace_num_sm)) 
  if ( .not. allocated(N_matrix_sm)) allocate(N_matrix_sm(subspace_num_sm,subspace_num_sm)) 


END SUBROUTINE setup_subspace_allocation_sm


SUBROUTINE read_subspace_wf_sm
  USE shell_model
  USE PARALLEL
 
  IMPLICIT NONE
  INTEGER :: channel_num, temp1, temp2, temp3, subspace_count
  INTEGER :: config, loop1, loop2
  character(LEN=50) :: inputfile, sm_file, str_temp, str_temp2
  REAL*8  :: real_temp, imag_temp

  do loop1 = 1, subspace_num_sm
     loop2 = loop1
     write(str_temp,"(i3)") loop2
     str_temp = adjustl(str_temp)
     sm_file=trim(str_temp)//'_sm.txt'
     if ( iam == 0 ) write(6,*) 'read_file' , sm_file

130  format (F20.15)
     open (223,file=sm_file, access="SEQUENTIAL")
     do config = 1, sm_config_num
         read(223,130) sm_evector_subspace(loop1,config) 
     end do
     close(223)
  end do


END SUBROUTINE read_subspace_wf_sm

SUBROUTINE get_N_matrix_sm
  USE shell_model
  USE PARALLEL

  IMPLICIT NONE
  INTEGER ::  bar,ket, loop1

  do bar = 1, subspace_num_sm
     do ket = 1, subspace_num_sm
        N_matrix_sm(bar,ket) = dot_product(sm_evector_subspace(bar,:),sm_evector_subspace(ket,:))
     end do 
  end do

 ! print *,N_matrix_sm

 
END SUBROUTINE get_N_matrix_sm

SUBROUTINE get_H_matrix_sm
  USE PARALLEL
  USE subspace
  USE single_particle_orbits
  USE one_body_operators
  USE constants
  USE configurations
  use shell_model
  use chiral_potentials

  IMPLICIT NONE
  real*8  ::  test, cutoff
  INTEGER :: p,q,r,s,temp_config, bar, ket, INFO, loop1
  INTEGER, allocatable ::  min_ev_index(:)

  if ( .not. allocated(temp_vector)) allocate(temp_vector(sm_config_num))

  temp_vector = 0


  do bar = 1, subspace_num_sm
     do ket = 1, subspace_num_sm
        do loop1 = 1, sm_config_num 
           temp_vector(loop1) = dot_product(sm_evector_subspace(bar,:),H_sm(:,loop1)) 
        end do 
        H_matrix_sm(bar,ket) = dot_product(temp_vector,sm_evector_subspace(ket,:))
     end do 
  end do
 
 ! do bar = 1, subspace_num_sm
 !    print *, H_matrix_sm(bar,:)
 ! end do 
END SUBROUTINE get_H_matrix_sm

SUBROUTINE print_N_H_matrix_sm
  use shell_model
  USE PARALLEL

  IMPLICIT NONE
  INTEGER :: channel, bar,ket, loop1, ij, ab
  character(LEN=50) :: inputfile, output_file, str_temp, str_temp2  
 
  output_file='H_matrix_sm.txt'
  open (227,file= output_file)
  do bar = 1, subspace_num_sm
   !  do ket = 1, subspace_num

120  format (64(F30.15,2x))  
     if ( iam == 0 ) write(227, 120) H_matrix_sm(bar,:)
   !  end do
  end do
  close(227)

  output_file='N_matrix_sm.txt'
  open (229,file= output_file)
  do bar = 1, subspace_num_sm
   !  do ket = 1, subspace_num
        if ( iam == 0 ) write(229,120) N_matrix_sm(bar,:)
   !  end do
  end do
  close(229)
 
END SUBROUTINE print_N_H_matrix_sm
