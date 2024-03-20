#!/vol/tcm30/rudenko/julia-1.7.3/bin/julia
using Printf
using DelimitedFiles
using LinearAlgebra
using Cubature

const hr_file="graphene_hr.dat"
const Emin=-5
const Emax=5
const μ = 11.3269
const Ndos=1000
const dos_smr=0.05
const nkx=128
const nky=128
const nkz=1
nk=nkx*nky*nkz
dE = (Emax-Emin)/Ndos


function w0gauss(x,n)::Float64

	if n==-99
		if abs(x)<36
		return 1 / ( 2 + exp(-x) + exp(+x) )
         	else 
		return 0
		end
	else
	arg = min(200,x^2)
	return exp(-arg)/sqrt(pi)
	end
end
	
###########
# Read HamR
###########
function read_hr()
nw=0
nr=0
head=0
degen=zeros(0)
open(hr_file) do file
    readline(file)
    nw = parse(Int,readline(file))
    nr = parse(Int,readline(file))
    data = read(file,String)
    sdegen = split(data)[1:nr]
    degen = parse.(Int,sdegen)
end
head = 3 + trunc(Int,nr/15+1)

 ham_data = zeros(nr*nw*nw,7)
 open(hr_file) do file
 ham_data = readdlm(file, Float64, skipstart=head)
 end

 HamR = reshape(ham_data[:,6],nw,nw,nr) + reshape(ham_data[:,7],nw,nw,nr)*1im

  Rx = reshape(ham_data[:,1],nw,nw,nr)[1,1,1:nr]
  Ry = reshape(ham_data[:,2],nw,nw,nr)[1,1,1:nr]
  Rz = reshape(ham_data[:,3],nw,nw,nr)[1,1,1:nr]
  R = hcat(Rx, Ry, Rz)
# K = Array{Float64,2}(undef,nk,3)

return (HamR, degen, R)
end
 
###########
# Diag HamR
###########
function diag_HamR()
#Ek = Array{Float16,2}(undef,nw,nk)
Ek = zeros(nw,nk)
K = zeros(nk,3)
for ikx = 1:nkx , iky = 1:nky, ikz=1:nkz
    ik = ikz + nkz*(iky-1) + nkz*nky*(ikx-1)
    K[ik,:] = [ (ikx-1)/nkx, (iky-1)/nky, (ikz-1)/nkz ]
    #
    HamK = zeros(nw,nw)*1im
    for ir = 1:nr
        rdotk = dot( K[ik,:], R[ir,:] )
        fac = exp( 2pi*rdotk*1im )
        HamK += fac * HamR[:,:,ir] / degen[ir]
    end
    Ek[:,ik] = eigvals(Hermitian(HamK))
#    for iv=1:nw
#	    @printf("%5d %12.12f %12.12f %12.12f %5d %12.12f \n", ik, K[ik,1], K[ik,2], K[ik,3], iv, Ek[iv,ik])
#    end
end

return Ek
end

###########
# Calc DOS
###########
function calc_dos()
#    function integrand(K)
#        dos = 0.0
#        Ek = diag_HamR(K)
#        for iw = 1:nw
#	    arg = (E - Ek[iw])/dos_smr
#	    dos += w0gauss(arg,-99)/dos_smr
#        end
#    return dos
#    end
#    result = hcubature( k -> integrand(k), [-0.5,-0.5,-0.5], [+0.5,+0.5,+0.5], reltol=1e-3, abstol=1e-5, maxevals=100)


dos = zeros(Ndos)
    wk = 1/nk
for ikx = 1:nkx , iky = 1:nky, ikz=1:nkz
    ik = ikz + nkz*(iky-1) + nkz*nky*(ikx-1)
    for iw = 1:nw
    for ie = 1:Ndos
	E = Emin + dE*ie
	arg = (E - Ek[iw,ik])/dos_smr
	dos[ie] += w0gauss(arg,-99)/dos_smr * wk
    end
    end
end
    return dos
end

###########
# Main part
###########

@time begin

HamR = read_hr()[1]
degen = read_hr()[2]
R = read_hr()[3]
nw = size(HamR,1)
nr = size(HamR,3)

Ek = diag_HamR()

dos = calc_dos()
for ie=1:Ndos
    E = Emin + dE*ie
    @printf("%12.12f %12.12f \n", E, dos[ie])
end

end

