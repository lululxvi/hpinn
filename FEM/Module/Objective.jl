#g=g_u(uvec)
f_target(x,Amp,h1,Lt,Ht) = Amp^2*((abs(x[1])<Lt/2)&&(abs(x[2]-h1/2)<Ht/2))

function g_u(u_vec;Amp,h1,Lt,Ht,U,V,dΩ_t)
  It(x) = f_target(x,Amp,h1,Lt,Ht)
  uh_t = FEFunction(U,u_vec)
  sum(∫(abs2(abs2(uh_t)-It))dΩ_t)
end

#uvec = u_pf(pf)
function u_pf(pf;x0,δ,Amp,P,Pf,β,η,flag_t,flag_f,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,U,V,dΩ,dΓ)
  if (flag_f)
      pfh = FEFunction(Pf,pf)
  else
      pfh = FEFunction(P,pf)
  end
  ph = (pf->Threshold(β,η,flag_t,pf))∘pfh
  A_mat = MatrixA(ph,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,U,V,dΩ)
  B_vec = MatrixB(x0,δ,2*π*Amp,V,dΩ,dΓ)
  u_vec = A_mat\B_vec
  u_vec
end

#pf = pf_p(p)
function pf_p(p;r,flag_f,P,Pf,Qf,dΩ,dΓ_d,tags,design_tag)
  pvec = p_vec(p,P,tags,design_tag)
  pf = Filter(pvec,r,flag_f,P,Pf,Qf,dΩ,dΓ_d)
  pf
end
# Chain Rule : dg/dp = dg/dg*dg/du*du/dpf*dpf/dp
# dg/du=dg/dg*dg/du
function rrule(::typeof(g_u),u_vec;Amp,h1,Lt,Ht,U,V,dΩ_t)
function g_pullback(dgdg)
  NO_FIELDS, dgdg*Dgdu(u_vec,Amp,h1,Lt,Ht,U,V,dΩ_t)
end
g_u(u_vec;Amp,h1,Lt,Ht,U,V,dΩ_t), g_pullback
end

function Dgdu(u_vec,Amp,h1,Lt,Ht,U,V,dΩ_t)
It(x) = f_target(x,Amp,h1,Lt,Ht)
uh_t = FEFunction(U,u_vec)
l_temp(du)=∫(4*uh_t*(abs2(uh_t)-It)*du)dΩ_t
assemble_vector(l_temp,V)
end

# dg/dpf=dg/du*du/dpf
function rrule(::typeof(u_pf),pf;x0,δ,Amp,P,Pf,β,η,flag_t,flag_f,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,U,V,dΩ,dΓ)
u_vec = u_pf(pf;x0,δ,Amp,P,Pf,β,η,flag_t,flag_f,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,U,V,dΩ,dΓ)
function u_pullback(dgdu)
  NO_FIELDS, Dgdpf(dgdu,u_vec,pf,P,Pf,β,η,flag_t,flag_f,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,U,V,dΩ)
end
u_vec, u_pullback
end

Dξdp(pf,ϵmin,ϵmax,β,η,flag_t)=(ϵmax-ϵmin)*(!flag_t+flag_t*β*(1.0-tanh(β*(pf-η))^2)/(tanh(β*η)+tanh(β*(1.0-η))))
dG(pfh,u,v,dp,ϵmin,ϵmax,k,β,η,flag_t) = real(k^2*((pf->Dξdp(pf,ϵmin,ϵmax,β,η,flag_t))∘pfh)*(v*u)*dp)

function Dgdpf(dgdu,u_vec,pf,P,Pf,β,η,flag_t,flag_f,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,U,V,dΩ)
  if (flag_f)
      pfh = FEFunction(Pf,pf)
      ph = (pf->Threshold(β,η,flag_t,pf))∘pfh
      A_mat = MatrixA(ph,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,U,V,dΩ)
      λ_vec = A_mat'\dgdu
      
      uh = FEFunction(U,u_vec)
      λh = FEFunction(V,conj(λ_vec))
      l_temp(dp) = ∫(dG(pfh,uh,λh,dp,ϵ1,ϵ2,k,β,η,flag_t))*dΩ
      dgdpf = assemble_vector(l_temp,Pf)
      return dgdpf
  else
      pfh = FEFunction(P,pf)
      ph = (pf->Threshold(β,η,flag_t,pf))∘pfh
      A_mat = MatrixA(ph,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,U,V,dΩ)
      λ_vec = A_mat'\dgdu
      
      uh = FEFunction(U,u_vec)
      λh = FEFunction(V,conj(λ_vec))
      l_temp(dp) = ∫(dG(pfh,uh,λh,dp,ϵ1,ϵ2,k,β,η,flag_t))*dΩ
      dgdpf = assemble_vector(l_temp,P)
      return dgdpf
  end
end

# dg/dp=dg/dpf*dpf/dp
function rrule(::typeof(pf_p),p;r,flag_f,P,Pf,Qf,dΩ,dΓ_d,tags,design_tag)
function pf_pullback(dgdpf)
  NO_FIELDS, Dgdp(dgdpf,p,r,flag_f,P,Pf,Qf,dΩ,dΓ_d,tags,design_tag)
end
pf_p(p;r,flag_f,P,Pf,Qf,dΩ,dΓ_d,tags,design_tag), pf_pullback
end

function Dgdp(dgdpf,p,r,flag_f,P,Pf,Qf,dΩ,dΓ_d,tags,design_tag)
np = length(p)
if (flag_f)
  A = assemble_matrix(Pf,Qf) do u, v
          ∫( a_f(r,u,v))dΩ
      end
  λvec = A'\dgdpf
  λh = FEFunction(Pf,λvec)
  l_temp(dp) = ∫(λh*dp)*dΩ
  return extract_design(assemble_vector(l_temp,P),np,tags,design_tag)
else
  return extract_design(dgdpf,np,tags,design_tag)
end
end

# Final objective function
function g_p(p::Vector;x0,δ,Amp,r,flag_f,P,Pf,Qf,β,η,flag_t,
      ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,h1,Lt,Ht,U,V,dΩ,dΓ,dΩ_t,dΓ_d,tags,design_tag)
  pf = pf_p(p;r,flag_f,P,Pf,Qf,dΩ,dΓ_d,tags,design_tag)
  u_vec=u_pf(pf;x0,δ,Amp,P,Pf,β,η,flag_t,flag_f,ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,U,V,dΩ,dΓ)
  g_u(u_vec;Amp,h1,Lt,Ht,U,V,dΩ_t)
end

function g_p(p::Vector,grad::Vector;x0,δ,Amp,r,flag_f,P,Pf,Qf,β,η,flag_t,
      ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,h1,Lt,Ht,U,V,dΩ,dΓ,dΩ_t,dΓ_d,tags,design_tag)
  if length(grad) > 0
      dgdp, = Zygote.gradient(p->g_p(p;x0,δ,Amp,r,flag_f,P,Pf,Qf,β,η,flag_t,
      ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,h1,Lt,Ht,U,V,dΩ,dΓ,dΩ_t,dΓ_d,tags,design_tag),p)
      grad[:] = dgdp
  end
  g_value = g_p(p;x0,δ,Amp,r,flag_f,P,Pf,Qf,β,η,flag_t,
      ϵ1,ϵ2,μ,σs,k,LHp,LHn,dpml,hd,h1,Lt,Ht,U,V,dΩ,dΓ,dΩ_t,dΓ_d,tags,design_tag)
  @show g_value
  return g_value
end