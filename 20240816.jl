using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, DifferentialEquations
using LinearAlgebra, Optim, DiffEqFlux, Flux, Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Flux, Plots, XLSX, Smoothers, Interpolations
using BSON: @save
using Base.Iterators: repeated
using Flux: onehotbatch, onecold, crossentropy, throttle
using LinearAlgebra, Random, ComponentArrays
using Lux, Random, Optimisers, Zygote
using SymbolicRegression, SymbolicUtils, MLJ, Interpolations
using DataFrames, GLM, StatsBase, LsqFit
using KissSmoothing, DataDrivenDiffEq, DataDrivenSparse
using ComplexityMeasures, Statistics
rng = Random.default_rng()
Random.seed!(714)


# data loading...
el = XLSX.readxlsx("Cases.xlsx")
sh = el["Sheet1"]
w = 21
tspan = (0.0, 357.0)

# Temperature T
AveT_origin = convert(Array{Float64}, sh["D2:D2507"])'
T_daily = []
for i in 1:358
    temp = AveT_origin[7*i-6] + AveT_origin[7*i-5] + AveT_origin[7*i-4] + AveT_origin[7*i-3] + AveT_origin[7*i-2] + AveT_origin[7*i-1] + AveT_origin[7*i]
    push!(T_daily, temp/7)
end

T_daily_new = convert(Array{Float64}, T_daily)[1:358]'
plot(T_daily_new')
AveT_al = hma(T_daily_new', w)' ./10
AveT = AveT_al[1:358]'
scatter(AveT')

x_ss = 0.0:1.0:357.0
T_ss = vec(AveT)
itp_cubic = cubic_spline_interpolation(x_ss, T_ss)
f_cubic(x) = itp_cubic(x)
T(x) = f_cubic(x)
scatter(AveT')
plot!(T.(x_ss))

# Average Humidity
AveH_origin = convert(Array{Float64}, sh["G2:G2507"])'
H_daily = []
for i in 1:358
    temp = AveH_origin[7*i-6] + AveH_origin[7*i-5] + AveH_origin[7*i-4] + AveH_origin[7*i-3] + AveH_origin[7*i-2] + AveH_origin[7*i-1] + AveH_origin[7*i]
    push!(H_daily, temp/7)
end
H_daily_new = convert(Array{Float64}, H_daily)[1:358]'

T_daily_n = T_daily_new ./ 10 .+ 273.15
e_s(t) = 6.11 * exp((2500000/461.52) * ((1/273.15) - (1/t)))
e(t, h) = e_s(t) * h / 100
mr(t, h) = (287.05/461.52) * (e(t, h)/(101325 - e(t, h)))
ah(t, h) = mr(t, h)/(1 + mr(t, h))
ah_daily_new = ah.(T_daily_n, H_daily_new)
plot(ah_daily_new')

AveH_al = hma(ah_daily_new[1:358], w)'
AveH = AveH_al[1:358]'
plot!(AveH_al')

H_ss = vec(AveH)
itp_cubic_H = cubic_spline_interpolation(x_ss, H_ss)
f_cubic_H(x) = itp_cubic_H(x)
AH(x) = f_cubic_H(x)
scatter(AveH')
plot!(AH.(x_ss))


# Environmental factors picture

y1 = [T_daily_new'./10 ah_daily_new']
y2 = [AveT' AveH']
scatter(y1, xlabel = "Weeks", ylabel = ["℃" "kg/kg"], label = ["T" "AH"], layout = (2, 1), w = 3)
plot!(y2, label = ["smooth T" "smooth AH"], layout = (2, 1), w = 1.5)
# savefig("Env.png")



# Cases
Cases_origin = convert(Array{Float64}, sh["B2:B2507"])'
Cases_daily = []
for i in 1:358
    temp = Cases_origin[7*i-6] + Cases_origin[7*i-5] + Cases_origin[7*i-4] + Cases_origin[7*i-3] + Cases_origin[7*i-2] + Cases_origin[7*i-1] + Cases_origin[7*i]
    push!(Cases_daily, temp)
end
Cases_daily_new = Cases_daily' ./ 7

Cases_al = hma(Cases_daily_new[:],21)'

Cases_temp = []
for i in 1:length(Cases_al)
    push!(Cases_temp, sum(Cases_al[1:i]))
end
Cases_total = convert(Array{Float64}, Cases_temp)'

scatter(Cases_daily_new', label = "Cases", w = 3, xlabel = "Weeks", ylabel = "Numbers")
plot!(Cases_al', w = 3, label = "Smoothing")
# savefig("Cases.png")


# NN
ss(x) = vcat(x[1:25], tanh.(x[26:40]), sin_cases.(x[41:end]))
sin_cases(x) = 65 * sin(0.068 * x)
relu(x) = max(0, x)
ann_caseses = Lux.Chain(
    Lux.Dense(3, 64, sin_cases), 
    ss,
    Lux.Dense(64, 1)
)
p_caseses, st_caseses = Lux.setup(rng, ann_caseses)
function model_caseses(du, u, p, t)
    Ann_caseses = ann_caseses([t, T.(t), AH.(t)], p, st_caseses)
    du[1] = -Ann_caseses[1][1] .+ 0.14 .* (12995900 - u[1] - u[2])
    du[2] = Ann_caseses[1][1] - (1/3) .* u[2]
    du[3] = Ann_caseses[1][1]
end
tspan = (0.0, 357.0)
prob_caseses = ODEProblem(model_caseses, [12995900; Cases_al[:, 1]; Cases_total[:, 1]], tspan, ComponentArray(p_caseses))

#training function
function train(θ)
    solve(prob_caseses, u0 = [12995900; Cases_al[:, 1]; Cases_total[:, 1]], p = θ, saveat = 1, sensealg = ForwardDiffSensitivity())
end
# train(p_cases)

#loss function
function loss(θ)
    pred = train(θ)[3, :]'
    sum(abs2, (Cases_total.- pred))
end
# loss(p_cases)

# callback
const losses_caseses = []

callback(θ, args...) = begin
    l = loss(θ)
    push!(losses_caseses, l)
    if length(losses_caseses) % 50 == 0
        println(losses_caseses[end])
    end
    false
end

# Optimization
pinit_caseses = ComponentArray(p_caseses)
adtype = Optimization.AutoZygote()
optf_caseses = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob_cases1es = Optimization.OptimizationProblem(optf_caseses, pinit_caseses)

@time result_cases1es = Optimization.solve(optprob_cases1es,
        OptimizationOptimisers.ADAM(0.031),
        callback = callback,
        maxiters = 7000)
# optprob_cases2 = remake(optprob_cases1, u0 = result_cases1.u)

# @time result_cases2 = Optimization.solve(optprob_cases2,
#         OptimizationOptimisers.ADAM(0.01),
#         callback = callback,
#         maxiters = 500,
#         allow_f_increases = false)

pfinal_caseses = result_cases1es.u
println(pfinal_caseses)


prob_cases2es = ODEProblem(model_caseses, [12995900; Cases_al[:, 1]; Cases_total[:, 1]], tspan, pfinal_caseses)
snnes = solve(prob_cases2es, Tsit5(), saveat = 1)
scatter(Cases_total')
plot!(snnes[3, :])


# Results comparison
resultes = []
for i in 1:1:357
    push!(resultes, snnes[3, i+1] - snnes[3, i])
end
scatter(Cases_al', w = 3, label = "Daily Cases", xlabel = "Weeks", ylabel = "Numbers")
plot!(resultes, w = 3, label = "Training Result")

# savefig("NN_t and real data.png")

const NN_resultes = []
new_tspan = collect(0.0:1.0:357.0)'
for t in new_tspan
    push!(NN_resultes, ann_caseses([t, T.(t), AH.(t)], pfinal_caseses, st_caseses)[1][1])
end
NN_reses = convert(Array{Float64}, NN_resultes)'
plot(NN_reses', label = "Training Result", w = 3, xlabel = "Weeks", ylabel = "Numbers")

# @save "./20240816_2.bson" ann_caseses
# psave_2 = collect(pfinal_cases)
# @save "./20240816_2_2.bson" psave_2
