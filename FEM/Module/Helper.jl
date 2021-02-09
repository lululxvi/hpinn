# Convert piece-wise constant p (design region) to pvec (whole domain)
function p_vec(p,P,tags,design_tag)
    pvec = zeros(num_free_dofs(P))
    pi = 0
    @assert length(tags)==num_free_dofs(P)
    for i=1:length(tags)
        if tags[i] == design_tag
            pi += 1
            pvec[i] = p[pi]
        end
    end
    pvec
end

# Extract the design region part from a whole vector
function extract_design(pvec,np,tags,design_tag)
    p_d = zeros(eltype(pvec),np)
    pi = 0
    @assert length(pvec)==length(tags)
    for i=1:length(tags)
        if tags[i] == design_tag
            pi += 1
            p_d[pi] = pvec[i]
        end
    end
    @assert np==pi
    p_d
end

# Gaussian Distribution function with center x0
function GaussianD(x,x0::AbstractArray,δ::AbstractArray)
    n=length(x)
    @assert (n==length(x0))&&(n==length(δ))
    δn = 1.0
    x_δ = 0.0
    for i=1:n
        δn *= √(2π)*δ[i]
        x_δ += ((x[i]-x0[i])/δ[i])^2
    end
    1.0/δn*exp(-x_δ/2.0)
end

# Gaussian Distribution function with center x0
function GaussianY(x,x0::AbstractArray,δ::AbstractArray)
    n=length(x)
    @assert (n==length(x0))&&(n==length(δ))
    δn = √(2π)*δ[2]
    x_δ = ((x[2]-x0[2])/δ[2])^2
    1.0/δn*exp(-x_δ/2.0)
end