SUBROUTINE setup_subspace_allocation
  USE single_particle_orbits
  USE constants
  use one_body_operators
  use t2_storage
  use configurations
  USE KSPACE
  use CHIRAL_POTENTIALS
  use subspace


  IMPLICIT NONE
  REAL*8 :: memory, denom
  INTEGER :: channel,bra,ket, dim1, dim2, a,b,i,j, nx2, ny2, nz2, tz2, bra2, ket2, channel2, loop1, loop2, loop3
  INTEGER :: channel_num

  channel_num = channels%number_hhpp_confs
  !if ( iam == 0 ) write(6,*) 'subspace_num' , subspace_num
  !if ( iam == 0 ) write(6,*) 'channels_num' , channels%number_hhpp_confs


  if ( .not. allocated(t2_subspace)) allocate( t2_subspace(subspace_num,channel_num) )
  if ( .not. allocated(l2_subspace)) allocate( l2_subspace(subspace_num,channel_num) )
  if ( .not. allocated(H_bar_subspace)) allocate( H_bar_subspace(subspace_num,channel_num) )
  if ( .not. allocated(kinetic_bar_subspace)) allocate( kinetic_bar_subspace(subspace_num,channel_num) )

  do loop1 = 1, subspace_num
     do channel   = 1, channel_num
   
        dim1 = size(  lookup_hhpp_configs(1,channel)%ival, 2)
        dim2 = size(  lookup_hhpp_configs(2,channel)%ival, 2)
   
        allocate( t2_subspace(loop1,channel)%val(dim2, dim1) )
        allocate( l2_subspace(loop1,channel)%val(dim2, dim1) )
        allocate( H_bar_subspace(loop1,channel)%val(dim2, dim1) )
        allocate( kinetic_bar_subspace(loop1,channel)%val(dim2, dim1) )
        
        t2_subspace(loop1,channel)%val = 0.0D0
        l2_subspace(loop1,channel)%val = 0.0D0
        H_bar_subspace(loop1,channel)%val = 0.0D0
        kinetic_bar_subspace(loop1,channel)%val = 0.0D0

     end do
  end do



!  if ( .not. allocated(t2_subspace)) allocate( t2_subspace(subspace_num, channel_num) )
!
!  do loop1 = 1, subspace_num
!
!     do channel   = 1, channel_num
!   
!        dim1 = size(  lookup_hhpp_configs(1,channel)%ival, 2)
!        dim2 = size(  lookup_hhpp_configs(2,channel)%ival, 2)
!        memory = memory + dble(dim1*dim2)*16./1.e9
!   
!        !if ( .not. allocated(t2_subspace(loop1,channel)%val)) allocate( t2_subspace(loop1,channel)%val(dim2, dim1) )
!        allocate( t2_subspace(loop1,channel)%val(dim2, dim1) )
!        
!        t2_subspace(loop1,channel)%val = (1.d0,1.d0)
!       ! do loop2 = 1, dim1
!       !    do loop3 = 1, dim2
!       !       t2_subspace(loop1,channel)%val(loop2,loop3) = (1.d0,1.d0)
!       !    end do
!       ! end do
!
!
!     end do
!  end do
!
!        dim1 = size(  lookup_hhpp_configs(1,channel)%ival, 2)
!        dim2 = size(  lookup_hhpp_configs(2,channel)%ival, 2)
!
!  if (iam == 0 ) write(6,*) 'dim1=', size(  lookup_hhpp_configs(1,1)%ival, 2)
!  if (iam == 0 ) write(6,*) 'dim2=', size(  lookup_hhpp_configs(2,1)%ival, 2)
!  if (iam == 0 ) write(6,*) 't2_subspace1=', t2_subspace(1,1)%val(1,4)
!  if (iam == 0 ) write(6,*) 't2_subspace2=', t2_subspace(1,1)%val(1,5)
!  if (iam == 0 ) write(6,*) 't2_subspace3=', t2_subspace(1,1)%val(1,6)
!  if (iam == 0 ) write(6,*) 'Total memory for t2 storage', 2.*memory, 'GByte'

END SUBROUTINE setup_subspace_allocation



SUBROUTINE read_subspace_matrix
  USE PARALLEL
  USE t2_storage
  USE subspace
  USE configurations

  IMPLICIT NONE
  INTEGER :: channel_num, temp1, temp2, temp3, subspace_count
  INTEGER :: channel, ij, ab, loop1, loop2
  character(LEN=50) :: inputfile, t2_file, str_temp, str_temp2  
  REAL*8  :: real_temp, imag_temp
 
  if ( iam == 0 ) write(6,*) 'read_subspace_matrix' 
  if ( iam == 0 ) write(6,*) 'subspace_num' , subspace_num
  if ( iam == 0 ) write(6,*) 'channels_num' , channels%number_hhpp_confs

  do loop1 = 1, subspace_num
     loop2 = loop1
     write(str_temp,"(i3)") loop2
     str_temp = adjustl(str_temp)
     t2_file=trim(str_temp)//'.txt'
     if ( iam == 0 ) write(6,*) 'read_file' , t2_file
 
     open (223,file=t2_file, access="SEQUENTIAL")
     do channel   = 1, channels%number_hhpp_confs
        do ij = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
           do ab = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)

              read(223,'(3(I4,1x),1x,2(g20.10,1x))') temp1, temp2, temp3, &
                   real_temp, imag_temp
              t2_subspace(loop1,channel)%val(ab,ij) = dcmplx(real_temp,imag_temp)    
              l2_subspace(loop1,channel)%val(ab,ij) = dcmplx(real_temp,imag_temp)    
           end do
        end do
     end do
     close(223)
  end do


!  t2_file='test.txt'
!
!  open (225,file=t2_file)
!  do channel   = 1, channels%number_hhpp_confs
!     do bra = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
!        do ket = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)
!
!           if ( iam == 0 ) write(225,'(3(I4,1x),1x,2(g20.10,1x))') channel, bra, ket, &
!                REAL(t2_subspace(1,channel)%val(ket,bra)), AIMAG(t2_subspace(1,channel)%val(ket,bra))
!        end do
!     end do
!  end do
!  close(225)

!  t2_file='test.txt'
!
!  open (225,file=t2_file)
!  do channel   = 1, channels%number_hhpp_confs
!     do bra = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
!        do ket = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)
!
!           if ( iam == 0 ) write(225,'(3(I4,1x),1x,2(g20.10,1x))') channel, bra, ket, &
!                REAL(vnn_hhpp(channel)%val(bra,ket)), AIMAG(vnn_hhpp(channel)%val(bra,ket))
!        end do
!     end do
!  end do
!  close(225)



END SUBROUTINE read_subspace_matrix


SUBROUTINE vacuum_expectation_value_H_bar
  USE PARALLEL
  USE subspace
  USE single_particle_orbits
  USE one_body_operators
  USE constants
  USE t2_storage
  USE configurations

!  USE single_particle_orbits
!  USE constants
!  use one_body_operators
!  use wave_functions
!  use chiral_potentials
!  use t2_storage
!  use ang_mom_functions
!  use subspace

  IMPLICIT NONE
  INTEGER :: channel, ij, ab, loop1, i, j,  nx2,ny2,nz2,tz2,sz2
  complex*16  :: H0, H3

  if ( .not. allocated(vacuum_H_bar)) allocate( vacuum_H_bar(subspace_num))
  vacuum_H_bar = 0.d0

  Ek = 0.d0 
  do i = 1, below_ef
     Ek = Ek + tkin(i,i)
  end do

  external_field_energy = e0 -Ek 
  if ( iam == 0 ) write(6,*) 'kinetic = ' , Ek
  if ( iam == 0 ) write(6,*) 'hf_energy = ' , e0
  if ( iam == 0 ) write(6,*) 'external_field_energy = ' , external_field_energy


  do loop1 = 1, subspace_num
     H0 = external_field_energy
     H3 = 0
     do channel   = 1, channels%number_hhpp_confs
        do ij = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
           do ab = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)
              H3 = H3 + 0.25d0 * vnn_hhpp(channel)%val(ij,ab) * t2_subspace(loop1,channel)%val(ab,ij)
           end do
        end do
     end do
     vacuum_H_bar(loop1) = H0 + H3
     !if ( iam == 0 ) write(6,*) 'vacuum_H_bar=' , vacuum_H_bar 
  end do
 

END SUBROUTINE vacuum_expectation_value_H_bar

 
SUBROUTINE H_bar_ijab
  USE PARALLEL
  USE CONSTANTS
  USE one_body_operators
  USE t2_storage
  USE subspace
  USE configurations
  USE diis_mod

  IMPLICIT NONE
  INTEGER :: count, itimes, ntimes, channel, bra, ket, ij, ab, i,j ,a ,b,loop1, loop2, channel_num
  complex*16 :: ener1, ener2, dener, d2f5d
  logical :: switch
  real*8  ::  startwtime , endwtime
  character(LEN=50) :: output_file

  channel_num = channels%number_hhpp_confs
  if ( iam == 0 ) write(6,*) 'subspace_num' , subspace_num
  if ( iam == 0 ) write(6,*) 'channels_num' , channel_num


  call setup_t_amplitudes

  !
  ! Allocate energy denominators
  !
  if ( .not. allocated(d2ij)) allocate( d2ij(1:below_ef))
  if ( .not. allocated(d2ab)) allocate( d2ab(below_ef+1:tot_orbs))
  d2ij = 0.d0; d2ab = 0.d0

  if ( iam == 0 ) write(6,*)  'fock_mtxall:' 
  do loop1=  1 , below_ef
     if ( iam == 0 ) write(6,"(20((1X,F10.5)))") REAL(fock_mtx(loop1,1:below_ef))
  end do




  do loop1 = 1, subspace_num

    
    ! set t2 amplitude for given subspace 
    do channel   = 1, channels%number_hhpp_confs
       do ij = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
          do ab = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)
             t2_ccm(channel)%val(ab,ij) = t2_subspace(loop1,channel)%val(ab,ij)
          end do
       end do
    end do
   
    !fock_mtx = fock_mtx_all 
 
    call t2_intermediate(0) 
    
    ! set H_bar_subsapce for given subspace 
    do channel   = 1, channels%number_hhpp_confs
       do ij = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
          i= lookup_hhpp_configs(1,channel)%ival(1,ij) 
          j = lookup_hhpp_configs(1,channel)%ival(2,ij) 

          do ab = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)
             a = lookup_hhpp_configs(2,channel)%ival(1,ab)
             b = lookup_hhpp_configs(2,channel)%ival(2,ab) 
 
             d2f5d= ( d2ij(i) + d2ij(j) - d2ab(a)-d2ab(b) )

             H_bar_subspace(loop1,channel)%val(ab,ij) = t2_ccm_eqn(channel)%val(ab,ij) 
             H_bar_subspace(loop1,channel)%val(ab,ij) = H_bar_subspace(loop1,channel)%val(ab,ij) - t2_subspace(loop1,channel)%val(ab,ij) *  d2f5d
          end do
       end do
    end do


!    fock_mtx = fock_mtx_1
    !fock_mtx = 0
!    if ( iam == 0 ) write(6,*) 'fock_mtx_1 =', fock_mtx_1 
!    call t2_intermediate(0) 
    
    ! set H_bar_subsapce for given subspace 
    do channel   = 1, channels%number_hhpp_confs
       do ij = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
          i= lookup_hhpp_configs(1,channel)%ival(1,ij) 
          j = lookup_hhpp_configs(1,channel)%ival(2,ij) 

         do ab = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)
             a = lookup_hhpp_configs(2,channel)%ival(1,ab)
             b = lookup_hhpp_configs(2,channel)%ival(2,ab) 
 
            kinetic_bar_subspace(loop1,channel)%val(ab,ij) = 0 
          end do
       end do
    end do

!  output_file='H_bar_mine.txt'
!  open (229,file= output_file)
!
!  if(iam ==0)write(229,*)  'H_bar:'
!  do channel = 1, channels%number_hhpp_confs
!     do ij = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
!        do ab = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)
!           if(iam ==0)write(229,*)  t2_ccm_eqn(channel)%val(ab,ij)
!        end do 
!     end do
!  end do
!  close(229)
!! print t2 amplitude
!  output_file='t2_mine.txt'
!  open (230,file= output_file)
!
!  if(iam ==0)write(230,*)  't2:'
!  do channel = 1, channels%number_hhpp_confs
!     do ij = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
!        do ab = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)
!           if(iam ==0)write(230,*)  t2_ccm(channel)%val(ab,ij)
!        end do 
!     end do
!  end do
!  close(230)





!  call ccd_energy_save(ener1)  

  end do  
  !ener1 = 0.d0


END SUBROUTINE H_bar_ijab

SUBROUTINE get_N_matrix  ! overlap of the subspace basis
  USE PARALLEL
  USE subspace
  USE t2_storage
  USE configurations


  IMPLICIT NONE
  INTEGER :: channel, bar,ket, loop1, ij, ab
  complex*16  :: N0, N3

  
  if ( .not. allocated(N_matrix)) allocate( N_matrix(subspace_num,subspace_num))

  do bar = 1, subspace_num
     do ket = 1, subspace_num
        N0 = 1
        N3 = 0
        do channel   = 1, channels%number_hhpp_confs
           do ij = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
              do ab = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)

                 N3 = N3 + 0.25d0 * l2_subspace(bar,channel)%val(ab,ij)*(-t2_subspace(bar,channel)%val(ab,ij) + t2_subspace(ket,channel)%val(ab,ij) )

              end do
           end do  
        end do

        N_matrix(bar,ket) = N0 + N3

     end do
  end do 


END SUBROUTINE get_N_matrix

SUBROUTINE get_H_matrix
  USE PARALLEL
  USE subspace
  USE t2_storage
  USE configurations

  IMPLICIT NONE
  INTEGER :: channel, bar,ket, loop1, ij, ab
  complex*16  :: H0, H3, K0, K3


  if ( .not. allocated(H_matrix)) allocate( H_matrix(subspace_num,subspace_num))
  if ( .not. allocated(K_matrix)) allocate( K_matrix(subspace_num,subspace_num))

  do bar = 1, subspace_num
     do ket = 1, subspace_num
        H0 = vacuum_H_bar(ket) * N_matrix(bar,ket) 
        K0 = Ek * N_matrix(bar,ket) 
        H3 = 0
        K3 = 0
        do channel   = 1, channels%number_hhpp_confs
           do ij = 1, size(  lookup_hhpp_configs(1,channel)%ival, 2)
              do ab = 1, size(  lookup_hhpp_configs(2,channel)%ival, 2)

                 H3 = H3 + 0.25d0 * l2_subspace(bar,channel)%val(ab,ij) * H_bar_subspace(ket,channel)%val(ab,ij) 
                 K3 = K3 + 0.25d0 * l2_subspace(bar,channel)%val(ab,ij) * kinetic_bar_subspace(ket,channel)%val(ab,ij)

              end do
           end do  
        end do

        if ( iam == 0 ) write(6,*) 'H0=', H0, ' H3=', H3
        if ( iam == 0 ) write(6,*) 'K0=', K0, ' K3=', K3
        H_matrix(bar,ket) = H0 + H3
        !K_matrix(bar,ket) = K0 + K3
        K_matrix(bar,ket) = K0 
     end do
  end do 


END SUBROUTINE get_H_matrix


SUBROUTINE print_N_H_K_matrix
  USE subspace
  USE PARALLEL 

  IMPLICIT NONE
  INTEGER :: channel, bar,ket, loop1, ij, ab
  character(LEN=50) :: inputfile, output_file, str_temp, str_temp2  
 
  output_file='H_matrix.txt'
  open (227,file= output_file)
  do bar = 1, subspace_num
   !  do ket = 1, subspace_num

120  format (64(F30.15,2x))  
     if ( iam == 0 ) write(227, 120) REAL(H_matrix(bar,:))
   !  end do
  end do
  close(227)

  output_file='K_matrix.txt'
  open (228,file= output_file)
  do bar = 1, subspace_num
   !  do ket = 1, subspace_num
        if ( iam == 0 ) write(228,120) REAL(K_matrix(bar,:))
   !  end do
  end do
  close(228)

  output_file='N_matrix.txt'
  open (229,file= output_file)
  do bar = 1, subspace_num
   !  do ket = 1, subspace_num
        if ( iam == 0 ) write(229,120) REAL(N_matrix(bar,:))
   !  end do
  end do
  close(229)


END SUBROUTINE print_N_H_K_matrix


