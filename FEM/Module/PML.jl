# PML coordinate streching functions
function s_PML(x,σs,Hd,k,LHp,LHn,dpml)
    σ = (x[2]<0)&&(x[2]+Hd>0) ? σs[2] : σs[1]
    xf = [x[1] x[2]]
    u = @. ifelse(xf>0 , xf-LHp , -xf-LHn)
    return @. ifelse(u > 0,  1-(1im*σ/k)*(u/dpml)^2, $(1.0+0im))
end

function Λ(x,σs,Hd,k,LHp,LHn,dpml)
    s_x,s_y = s_PML(x,σs,Hd,k,LHp,LHn,dpml)
    return TensorValue(1/s_x,0,0,1/s_y) 
end