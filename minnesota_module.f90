module minnesota_potentials

  implicit none

contains

  function minnesota(p,q,r,s) result(res)
    USE single_particle_orbits
    USE constants
    use chiral_constants
    use ang_mom_functions, only : tjs  

    implicit none
    INTEGER, INTENT(IN) :: p,q,r,s 
    INTEGER :: m1,m2,m3,m4, spin, iph, t1,t2,t3,t4, Tiso 
    INTEGER :: nx1, ny1, nz1, nx2, ny2, nz2, nx3, ny3, nz3, nx4, ny4, nz4, mt1, mt2, mt3, mt4  
    integer :: impi 
    REAL*8 :: k1(3), k2(3), k3(3), k4(3), kmean(3) 
    REAL*8 :: qtrans(3), prel(3), pprel(3), qxk(3)
    REAL*8 :: q2, p2, kmean2, qabs, pp2, cg1, cg2 ,qq
    REAL*8 :: delta, nucleon_mass, relativity_factor
    complex*16 :: res
    COMPLEX*16 :: vlo, vnlo, vnnlo, vn3lo, vdir 
    COMPLEX*16 :: cont_lo, cont_nlo, cont_n3lo, v1pe_tiso(0:1)

    res = 0.d0

    nx1 = all_orbit%nx(p)
    ny1 = all_orbit%ny(p)
    nz1 = all_orbit%nz(p)
    nx2 = all_orbit%nx(q)
    ny2 = all_orbit%ny(q)
    nz2 = all_orbit%nz(q)
    nx3 = all_orbit%nx(r)
    ny3 = all_orbit%ny(r)
    nz3 = all_orbit%nz(r)
    nx4 = all_orbit%nx(s)
    ny4 = all_orbit%ny(s)
    nz4 = all_orbit%nz(s)
    ! 
    ! Conservation of linear momentum
    !
    if ( nx1 + nx2 /= nx3 + nx4 ) return
    if ( ny1 + ny2 /= ny3 + ny4 ) return
    if ( nz1 + nz2 /= nz3 + nz4 ) return

    k1(1) = all_orbit%kx(p)
    k1(2) = all_orbit%ky(p)
    k1(3) = all_orbit%kz(p)
    k2(1) = all_orbit%kx(q)
    k2(2) = all_orbit%ky(q)
    k2(3) = all_orbit%kz(q)
    k3(1) = all_orbit%kx(r)
    k3(2) = all_orbit%ky(r)
    k3(3) = all_orbit%kz(r)
    k4(1) = all_orbit%kx(s)
    k4(2) = all_orbit%ky(s)
    k4(3) = all_orbit%kz(s)

    ! 
    ! conservation of isospin 
    !
    if ( all_orbit%itzp(p) + all_orbit%itzp(q) /= all_orbit%itzp(r) + all_orbit%itzp(s) ) return

    m1 = all_orbit%szp(p)
    m2 = all_orbit%szp(q)
    m3 = all_orbit%szp(r)
    m4 = all_orbit%szp(s)

    t1 = all_orbit%itzp(p)
    t2 = all_orbit%itzp(q)
    t3 = all_orbit%itzp(r)
    t4 = all_orbit%itzp(s)

    ! 
    ! RELATIVE MOMENTA <prel |v| pprel > 
    ! 
    prel  = 0.5d0*(k1-k2)
    pprel = 0.5d0*(k3-k4)
    p2  = sum(prel*prel)
    pp2 = sum(pprel*pprel)

    !
    ! momentum transfer 
    !
    qtrans = prel - pprel
    q2 = sum(qtrans*qtrans)
    qabs = dsqrt(q2)


    res = res + 200.d0  / lx**3.d0 * (pi/1.487d0)**(1.5d0) * exp(-q2/(4.d0*1.487d0));
    res = res + -91.85d0  / lx**3.d0 * (pi/0.465d0)**(1.5d0) * exp(-q2/(4.d0*0.465d0));
    res = 0.5d0 * res * (kronecker(m1,m3) * kronecker(m2,m4) - kronecker(m1,m4) * kronecker(m2,m3))

  end function minnesota

  function kronecker(a,b) result(delta)
    implicit none
    INTEGER, INTENT(IN) :: a, b
    INTEGER :: delta
  
    if (a == b) then
       delta = 1
    else 
       delta = 0
    end if
  end function kronecker
end module minnesota_potentials
