# Define the theoretic model
################### Assemble matrix and vector #####################
# Weak form of the Helmholtz equation : 
# Hz: a(p,u,v)=Λ⋅∇v⋅ξ(p)Λ⋅∇u-k²μv⋅u
# Material distribution
#ξ0(x,ϵ1,ϵ2) = x[2]>0 ? 1/ϵ1 : 1/ϵ2
#ξd(p,ϵmin,ϵmax)= 1/(ϵmin + (ϵmax-ϵmin)*p) - 1/ϵmin # in the design region
#a0(u,v,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml) = ((x->Λ(x,σs,k,LHp,LHn,dpml))⋅∇(v))⊙((x->ξ0(x,ϵ1,ϵ2))*((x->Λ(x,σs,k,LHp,LHn,dpml))⋅∇(u))) - k^2*μ*(v*u)
#a_d(u,v,ph,k,ϵ1,ϵd) = ∇(v)⊙(((p -> ξd(p, ϵ1, ϵd))∘ph)*∇(u))
# Ez: a(p,u,v)=1/μ*Λ⋅∇v⋅Λ⋅∇u-k²ξ(p)uv
ξd(p,ϵmin,ϵmax)= ϵmin+(ϵmax-ϵmin)*p # in the design region
a0(u,v,ph,ϵ1,ϵ2,μ,σs,k,LHp,LHn,Hd,dpml) = ((x->Λ(x,σs,Hd,k,LHp,LHn,dpml))⋅∇(v))⊙((1/μ)*((x->Λ(x,σs,Hd,k,LHp,LHn,dpml))⋅∇(u))) - k^2*(((p -> ξd(p, ϵ1, ϵ2))∘ph)*(v*u))
#a_d(u,v,ph,k,ϵ1,ϵd) = -k^2*((p -> ξd(p, ϵ1, ϵd))∘ph)*(v*u)
# Source term (Gaussian point source approximation at center)

#b(v,x0,δ) = v*(x->GaussianD(x,x0,δ))


function MatrixA(ph,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,Hd,U,V,dΩ)
    # Assemble the matrix
    return assemble_matrix(U,V) do u,v
        ∫( a0(u,v,ph,ϵ1,ϵ2,μ,σs,k,LHp,LHn,Hd,dpml) )dΩ
    end
end

function MatrixB(x0,δ,Amp,V,dΩ,dΓ)
    l_temp(v) = ∫( Amp*v*x->GaussianY(x,x0,δ) )*dΩ+∫( 0*v )*dΓ
    #op = AffineFEOperator((u,v)->(∫(u*v)*dΩ),l_temp,U,V)
    #op = FESource(l_temp,U,V)
    assemble_vector(l_temp,V)
end
