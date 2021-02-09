################### Filter and Threshold #####################
# pf = Filter(p)
a_f(r,u,v) = r^2*(∇(v)⊙∇(u))+v⊙u
function Filter(pvec,r,flag_f::Bool,P,Pf,Qf,dΩ,dΓ)
    if (flag_f)
        ph = FEFunction(P,pvec)
        op = AffineFEOperator(Pf,Qf) do u, v
            ∫( a_f(r,u,v))dΩ, ∫( v*ph )dΩ,∫( 0*v )dΓ
          end
        pf = solve(op)
        return get_free_values(pf)
    else
        return pvec
    end
end

# Threshold function
Threshold(β,η,flag_t::Bool,pf) = flag_t==false ? pf : ((tanh(β*η)+tanh(β*(pf-η)))/(tanh(β*η)+tanh(β*(1.0-η))))