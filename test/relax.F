C
C A subroutine to calculate the Jacobi relaxation in 2D.
C
      subroutine relax2D(phi, output, rhs, m, n, dx, dy, w)

      integer m, n
      real*8 phi(m,n), output(m,n), rhs(m,n)
      real*8 dx, dy, w, a

      do j = 2, n-1
        do i = 2, m-1
          a = (rhs(i,j) - ((phi(i-1,j) + phi(i+1,j)) / (dx*dx)
     &      + phi(i,j-1) + phi(i,j+1) / (dy*dy)))
     &      / (2.0/dx/dx + 2.0/dy/dy)
          output(i,j) = a * w + phi(i,j) * (1-w)
        enddo
      enddo

      return
      end
