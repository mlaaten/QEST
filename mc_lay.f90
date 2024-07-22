      FUNCTION g_3d_karman(k,a,nu,theta)
! *** computes NORMALIZED angular dependent scattering coefficient for
! *** 3D Von Karman type random medium, normalzation is such that
! *** g(theta=0) = 1
      IMPLICIT NONE
      DOUBLE PRECISION :: g_3d_karman, theta, a, nu, k
      g_3d_karman=(1.0+(2.0*a*k*SIN(theta/2.0))**2)**(-nu-1.5)
      RETURN
      END
      
      FUNCTION gammln(xx)
      INTEGER :: j
      DOUBLE PRECISION :: ser, stp, tmp, x, y, cof(6), gammln, xx
      SAVE cof,stp
      DATA cof,stp/76.18009172947146,-86.50532032941677, &
      & 24.01409824083091,-1.231739572450155,.1208650973866179d-2, &
      & -.5395239384953d-5,2.5066282746310005/
      x=xx
      y=x
      tmp=x+5.5
      tmp=(x+0.5)*LOG(tmp)-tmp
      ser=1.000000000190015
      DO 11 j=1,6
        y=y+1.0
        ser=ser+cof(j)/y
11    CONTINUE
      gammln=tmp+LOG(stp*ser/x)
      RETURN
      END      
 
      FUNCTION g0_3d_karman(k,a,eps,nu)
      IMPLICIT NONE
      DOUBLE PRECISION :: pi
      DOUBLE PRECISION :: g0_3d_karman, a, eps, k, nu, gammln
      pi=4.0*ATAN(1.0)
      g0_3d_karman = 2.0 * SQRT(pi) &
      & * EXP(DBLE(gammln(nu+1.5))) &
      & / EXP(DBLE(gammln(nu))) / (nu+0.5) * (k*eps)**2 * a &
      & * (1.0d0-((2.0*a*k)**2+1.0)**(-nu-0.5))
      RETURN
      END
          
      FUNCTION gtr_3d_karman(k,a,eps,nu)
      IMPLICIT NONE
      DOUBLE PRECISION :: pi
      DOUBLE PRECISION :: gtr_3d_karman, a, eps, k, nu, gammln
      pi=4.0*ATAN(1.0)
      gtr_3d_karman=SQRT(pi)*(eps)**2 &
      & * EXP(DBLE(gammln(nu+1.5))) &
      & / EXP(DBLE(gammln(nu))) &
      & / a / (nu+0.5) / (nu-0.5) &
      & *((nu-0.5)*(1+4*a**2*k**2)**(-nu-0.5) &
      &  -(nu+0.5)*(1+4*a**2*k**2)**(-nu+0.5) + 1.0)
      RETURN
      END      
      
      
 
      SUBROUTINE MC_LAY_3D_NISO(k_depth, a_depth, eps_depth, nu_depth, B_depth,Z0,Npar,Rho,Zz,nrec, &
                            & Dt,dr,v_depth,v_vel,rho_depth,att_depth,rdm_sc_l,Ntime,E_dp,tt,s)
                 ! Inputs: k_depth, a_depth, eps_depth, nu_depth, B_depth,Z0,Npar,Rho,Zz,nrec,Dt,dr,v_depth,v_vel,rho_depth,att_depth,rdm_sc_l,Ntime
                 ! Output: E_dp tt, s

! compile with: python -m numpy.f2py -c mc_lay.f90 -m mc_lay
! *** computes energy in an acoustic scattering layer over a
! *** homogeneous halfspace
! *** assuming isotropic scattering and isotropic source radiation
! *** Ulrich Wegler, 30.09.2003
! *** corrected version, Ulrich Wegler, 02.07.2004
! *** edited version, Marcel van Laaten, 12.08.2021
!   
      IMPLICIT NONE
      
!*--MC_LAY_3D_ISO12
!
! *** input parameter:
      INTEGER :: Npar
      INTEGER :: Ntime
      INTEGER :: nrec
      INTEGER :: rdm_sc_l
      DOUBLE PRECISION :: Z0
      DOUBLE PRECISION :: Dt
      DOUBLE PRECISION :: dr
      DOUBLE PRECISION, dimension(:) :: Rho
      DOUBLE PRECISION, dimension(:) :: Zz
      DOUBLE PRECISION, dimension(:) :: v_depth
      DOUBLE PRECISION, dimension(:) :: att_depth
      DOUBLE PRECISION, dimension(:) :: v_vel
      DOUBLE PRECISION, dimension(:) :: rho_depth
      DOUBLE PRECISION, dimension(:) :: B_depth
      DOUBLE PRECISION, dimension(:) :: a_depth
      DOUBLE PRECISION, dimension(:) :: eps_depth
      DOUBLE PRECISION, dimension(:) :: nu_depth
      DOUBLE PRECISION, dimension(:) :: k_depth
     
!    
! *** output parameters:
      DOUBLE PRECISION, dimension(nrec, Ntime) :: E_dp
!
! *** set parameters:
      DOUBLE PRECISION :: PI
      DOUBLE PRECISION :: PI_TIMES_2
!
! *** declaration of variables:
      INTEGER :: ipar
      INTEGER :: itime
      INTEGER :: dp
      INTEGER :: indi,indi2, indi_sc  
      INTEGER :: istat
      INTEGER :: ii, iii, iiii
      DOUBLE PRECISION, dimension(:), allocatable :: rdm_sc
      INTEGER :: row
      DOUBLE PRECISION, dimension(nrec, Ntime) :: e_int
      DOUBLE PRECISION, dimension(Ntime) :: tt
      DOUBLE PRECISION, dimension(nrec) :: s
      DOUBLE PRECISION, dimension(SIZE(v_depth)) :: G0_depth
      DOUBLE PRECISION, dimension(:), allocatable :: rdm_sc_i
      DOUBLE PRECISION :: x(3)
      DOUBLE PRECISION :: v(3), v_t(3)
      DOUBLE PRECISION :: W
      DOUBLE PRECISION :: phi, phi_s
      DOUBLE PRECISION :: the, the_, the_s
      DOUBLE PRECISION :: the2_
      DOUBLE PRECISION :: rdm(Npar*10)
      DOUBLE PRECISION :: u2
      DOUBLE PRECISION :: v_depthval     
      DOUBLE PRECISION :: att_depthval     
      DOUBLE PRECISION :: V1, V2
      DOUBLE PRECISION :: h
      DOUBLE PRECISION :: B, G0
      DOUBLE PRECISION :: Z1
      DOUBLE PRECISION :: rat_bef 
      DOUBLE PRECISION :: remp
      DOUBLE PRECISION :: rho1, rho2
      DOUBLE PRECISION :: reflec, delta
      DOUBLE PRECISION :: g0_3d_karman
      


      !f2py intent(in) :: k_depth,a_depth,eps_depth,nu_depth,B_depth,Z0,Npar,Rho,Zz,nrec,Dt,dr,v_depth,v_vel,rho_depth,att_depth,rdm_sc_l,Ntime
      !f2py intent(out) :: E_dp, tt, s





! *** compute more parameters:
      CALL random_number(rdm)
      
      open(1,file='scat.dat', status='old',action='read')
      allocate(rdm_sc(rdm_sc_l))
      DO row = 1, rdm_sc_l
          read(1,*) rdm_sc(row)
      ENDDO 
      
      allocate(rdm_sc_i(rdm_sc_l))
      CALL random_number(rdm_sc_i)
      rdm_sc_i=rdm_sc_i*SIZE(rdm_sc)
      
      PI=4.0*ATAN(1.0)
      PI_TIMES_2=8.0*ATAN(1.0)

      DO iii = 1, SIZE(v_depth)
         G0_depth(iii)=g0_3d_karman(k_depth(iii),a_depth(iii),eps_depth(iii),nu_depth(iii)) !total scattering coefficient
      ENDDO
      DO istat = 1, nrec
         !s(istat) = 2.*PI**2*Rho(istat)*dr**2
         !s(istat) = s(istat)/2.0D0
         h=dr-Zz(istat)
         s(istat) = 2.*PI*Rho(istat)*(dr**2*ACOS(1.-h/dr)-(dr-h)*SQRT(2.*dr*h*h**2))
! *** init dummy variable and energy field
         DO itime = 1 , Ntime
             e_int(istat,itime) = 0.
         ENDDO
      ENDDO    
! *** loop for all particles
      ii = 1
      iiii = 1
      DO ipar = 1 , Npar
         W = 1.
! ****** initial position at origin and initial velocity in random direction
         x(1) = 0.
         x(2) = 0.
         x(3) = Z0
         phi = PI_TIMES_2*rdm(ii)
         u2 = 1. - 2.*rdm(ii+1)
         ii=ii+2
         IF ( ii.gt.SIZE(rdm) ) THEN
            CALL random_number(rdm)
            ii=1
         ENDIF
         the = ACOS(u2)       
         v_depthval = MAXVAL(v_depth, mask=(v_depth <= x(3)))
         indi = MINLOC(v_depth, dim=1, mask=(v_depth >= v_depthval)) 
         att_depthval = MAXVAL(att_depth, mask=(att_depth <= x(3)))
         indi2 = MINLOC(att_depth, dim=1, mask=(att_depth >= att_depthval)) 
         G0=G0_depth(indi2)
         B=B_depth(indi2)    
         V1 = v_vel(indi)
         rho1 = rho_depth(indi)
         v(1) = V1*SIN(the)*COS(phi)
         v(2) = V1*SIN(the)*SIN(phi)
         v(3) = V1*COS(the)
! ****** loop for time
         DO itime = 1 , Ntime
! ********* absorption
            W = W * EXP(-B*Dt) 
! ********* free motion of particle
            x(1) = x(1) + v(1)*Dt
            x(2) = x(2) + v(2)*Dt
            x(3) = x(3) + v(3)*Dt 
! ********* reflection at the free surface
            IF ( x(3).LE.0.0D0 ) THEN
               x(3) = -x(3)
               v(3) = -v(3)
               the = PI - the
               GOTO 300
            ENDIF 
! ********* reflection and loss at the bottom of the layer
            DO dp = 1, SIZE(v_depth)
               Z1 = v_depth(dp)
               IF ((x(3)-v(3)*Dt).LT.0.0) GOTO 300
               IF ((x(3)-v(3)*Dt.LE.Z1.AND.x(3).GE.Z1) &
               & .OR.(x(3)-v(3)*Dt.GE.Z1.AND.x(3).LE.Z1)) THEN           
                  IF ( the.GE.PI/2 ) THEN                     
                     the_ = PI - the
                  ELSE
                     the_ = the
                  ENDIF 
                  v_depthval = MAXVAL(v_depth, mask=(v_depth <= x(3)))
                  indi = MINLOC(v_depth, dim=1, mask=(v_depth >= v_depthval))                           
                  V2 = v_vel(indi)       
                  rho2 = rho_depth(indi)   
                  the2_ = ASIN((SIN(the_)*V2)/V1)       
                  IF (isnan(the2_)) THEN !total reflection 
                     x(3) = 2*Z1 - x(3)
                     v(3) = -v(3)
                     the = PI - the
                  ELSE
                     delta = rho1 * V1 * COS(the_) + rho2 * V2 * COS(the2_)
                     reflec = ((rho1*V1*COS(the_)-rho2*V2*COS(the2_))/delta)**2
                     IF ( rdm(ii).GT.ABS(reflec) ) THEN !transmission
                        x(1) = x(1) - v(1)*Dt
                        x(2) = x(2) - v(2)*Dt
                        x(3) = x(3) - v(3)*Dt                 
                        rat_bef = ABS(Z1 - x(3)) / ABS(v(3)*Dt)                    
                        x(1) = x(1) + v(1)*rat_bef*Dt
                        x(2) = x(2) + v(2)*rat_bef*Dt
                        x(3) = x(3) + v(3)*rat_bef*Dt  
                        remp=(1-rat_bef)*SQRT((v(1)*Dt)**2 &
                           & +(v(2)*Dt)**2+(v(3)*Dt)**2)
                        remp = (remp / V1) * V2         
                        IF ( v(3).GE.0 ) THEN                   
                           the = the2_
                        ELSE
                           the = PI - the2_
                        ENDIF               
                        v_t(1) = remp*SIN(the)*COS(phi) 
                        v_t(2) = remp*SIN(the)*SIN(phi)
                        v_t(3) = remp*COS(the)
                        x(1) = x(1) + v_t(1)
                        x(2) = x(2) + v_t(2)
                        x(3) = x(3) + v_t(3)
                        v_depthval = MAXVAL(v_depth, mask=(v_depth <= x(3)))
                        indi = MINLOC(v_depth, dim=1, mask=(v_depth >= v_depthval)) 
                        V1 = v_vel(indi)
                        rho1 = rho_depth(indi)
                        v(1) = V1*SIN(the)*COS(phi)
                        v(2) = V1*SIN(the)*SIN(phi)
                        v(3) = V1*COS(the) 
                        att_depthval = MAXVAL(att_depth, mask=(att_depth <= x(3)))
                        indi2 = MINLOC(att_depth, dim=1, mask=(att_depth >= att_depthval))      
                        B=B_depth(indi2)    
                        G0=G0_depth(indi2)
                        GOTO 300
                     ELSE !reflection
                        x(3) = 2*Z1 - x(3)
                        v(3) = -v(3)
                        the = PI - the
                        GOTO 300
                     ENDIF
                     ii=ii+1
                     IF ( ii.gt.SIZE(rdm) ) ii=1
                  ENDIF
               ENDIF
            ENDDO
! ********* nonisotropic scattering in new direction           
 300        IF ( rdm(ii).LT.(V1*G0*Dt) ) THEN  
                phi_s = PI_TIMES_2*rdm(ii+1)   
                ii=ii+2    
                IF ( ii.gt.SIZE(rdm) ) ii=1   
                
                indi_sc = NINT(rdm_sc_i(iiii))
                IF (indi_sc.eq.0 ) indi_sc=1
                iiii=iiii+1
                IF ( iiii.gt.SIZE(rdm_sc_i) ) THEN
                   CALL random_number(rdm_sc_i)
                   rdm_sc_i=rdm_sc_i*SIZE(rdm_sc)
                   iiii=1
                ENDIF
                the_s = rdm_sc(indi_sc)
                
                !the_s = rdm_sc(iiii)
                !iiii=iiii+1
                !IF ( iiii.gt.SIZE(rdm_sc) ) iiii=1   
! *************** compute new direction of propagation in absolute coordinate system (Euler-rotation)
                v(1)=V1*(SIN(the_s)*COS(phi_s)*COS(the)*COS(phi) & 
	           & + COS(the_s)*SIN(the)*COS(phi) & 
	           & - SIN(the_s)*SIN(phi_s)*SIN(phi)) !new velocity, x-component
                v(2)=V1*(SIN(the_s)*COS(phi_s)*COS(the)*SIN(phi) &
                   & + COS(the_s)*SIN(the)*SIN(phi) &
                   & + SIN(the_s)*SIN(phi_s)*COS(phi))  !new velocity, y-component
                v(3)=V1*(-SIN(the_s)*COS(phi_s)*SIN(the) &
	           & + COS(the_s)*COS(the)) !new velocity, z-component
                the=ATAN(SQRT(v(1)**2+v(2)**2)/v(3)) !new direction of motion, theta-angle    
                IF (the.lt.0) the=pi+the !lower halfspace
                phi=ATAN(v(2)/v(1)) 
                IF (v(1).lt.0.and.v(2).ge.0) phi=phi+pi !2. quadrant
                IF (v(1).lt.0.and.v(2).lt.0) phi=phi+pi  !3. quadrant
                IF (v(1).ge.0.and.v(2).lt.0) phi=phi+2*pi !4. quadrant
            ENDIF
            ii=ii+1
            IF ( ii.gt.SIZE(rdm) ) ii=1   
! ********* collect particles
            DO istat = 1, nrec
                IF((SQRT(x(1)**2+x(2)**2)-Rho(istat))**2 &
          &         +(x(3)-Zz(istat))**2.LE.dr**2) THEN 
                   e_int(istat,itime) = e_int(istat,itime) + W
                ENDIF
            ENDDO
         ENDDO
      ENDDO
      ! ****** normalize energy
      DO istat = 1, nrec
         DO itime = 1 , Ntime
            tt(itime) = itime*Dt
            IF ( e_int(istat,itime).EQ.0.0D0 ) THEN
                !E_dp(istat,itime) = 1/(s(istat)*Npar)
                E_dp(istat,itime) = 1D-99
            ELSE
                !E_dp(istat,itime) = e_int(istat,itime)/(s(istat)*Npar)
                E_dp(istat,itime) = e_int(istat,itime)
            ENDIF
         ENDDO
      ENDDO
! *** end of programm
      return
      deallocate (rdm_sc)
      deallocate (rdm_sc_i)
      close(1)
      END SUBROUTINE

