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
savefig("Env.png")



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
ann_cases = Lux.Chain(
    Lux.Dense(1, 64, sin_cases), 
    ss,
    Lux.Dense(64, 1)
)
p_cases, st_cases = Lux.setup(rng, ann_cases)
function model_cases(du, u, p, t)
    Ann_cases = ann_cases([t], p, st_cases)
    du[1] = -Ann_cases[1][1] .+ 0.14 .* (12995900 - u[1] - u[2])
    du[2] = Ann_cases[1][1] - (1/3) .* u[2]
    du[3] = Ann_cases[1][1]
end
tspan = (0.0, 357.0)
prob_cases = ODEProblem(model_cases, [12995900; Cases_al[:, 1]; Cases_total[:, 1]], tspan, ComponentArray(p_cases))

#training function
function train(θ)
    solve(prob_cases, u0 = [12995900; Cases_al[:, 1]; Cases_total[:, 1]], p = θ, saveat = 1, sensealg = ForwardDiffSensitivity())
end
# train(p_cases)

#loss function
function loss(θ)
    pred = train(θ)[3, :]'
    sum(abs2, (Cases_total.- pred))
end
# loss(p_cases)

# callback
const losses_cases = []

callback(θ, args...) = begin
    l = loss(θ)
    push!(losses_cases, l)
    if length(losses_cases) % 50 == 0
        println(losses_cases[end])
    end
    false
end

# Optimization
pinit_cases = ComponentArray(p_cases)
adtype = Optimization.AutoZygote()
optf_cases = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob_cases1 = Optimization.OptimizationProblem(optf_cases, pinit_cases)

@time result_cases1 = Optimization.solve(optprob_cases1,
        OptimizationOptimisers.ADAM(0.031),
        callback = callback,
        maxiters = 7000)
# optprob_cases2 = remake(optprob_cases1, u0 = result_cases1.u)

# @time result_cases2 = Optimization.solve(optprob_cases2,
#         OptimizationOptimisers.ADAM(0.01),
#         callback = callback,
#         maxiters = 500,
#         allow_f_increases = false)

pfinal_cases = result_cases1.u
println(pfinal_cases)


prob_cases2 = ODEProblem(model_cases, [12995900; Cases_al[:, 1]; Cases_total[:, 1]], tspan, pfinal_cases)
snn = solve(prob_cases2, Tsit5(), saveat = 1)
scatter(Cases_total')
plot!(snn[3, :])

# @save "./ann_1.bson" ann_cases
# psave_1 = collect(pfinal_cases)
# @save "./ann_1_1.bson" psave_1

# Results comparison
result = []
for i in 1:1:357
    push!(result, snn[3, i+1] - snn[3, i])
end
scatter(Cases_al', w = 3, label = "Daily Cases", xlabel = "Weeks", ylabel = "Numbers")
plot!(result, w = 3, label = "Training Result")

# savefig("NN_t and real data.png")

const NN_result = []
new_tspan = collect(0.0:1.0:357.0)'
for t in new_tspan
    push!(NN_result, ann_cases([t], pfinal_cases, st_cases)[1][1])
end
NN_res = convert(Array{Float64}, NN_result)'

ir(t) = ann_cases([t], pfinal_cases, st_cases)[1][1]
plot(ir.(new_tspan)')
function RecoverDE(du, u, p, t)
    du[1] = -ir.(t) + 0.14 * (12995900 - u[1] - u[2])
    du[2] = ir.(t) - (1/3) * u[2]
    du[3] = ir.(t)
end
prob_re = ODEProblem(RecoverDE, [12995900; Cases_al[:, 1]; Cases_total[:, 1]], tspan)
sol_re = solve(prob_re, saveat = 1, sensealg = ForwardDiffSensitivity())


# # # SymbolicRegression Part
S = sol_re[1, :]
I = sol_re[2, :]
X1 = convert(Array{Float64}, vcat(S'./12995900, I'))
options = SymbolicRegression.Options(
    # binary_operators = (+, -, *,),
    binary_operators = (*,),
    npopulations = 50,
    ncyclesperiteration = 300,
    alpha = 0.100000f0,
    maxsize = 10,
)
hallOfFrame_1 = EquationSearch(X1, NN_res, niterations = 100, options = options)


# SINDy Part
y = NN_res .* 12995900
problem_1 = DirectDataDrivenProblem(X1, y)
@variables u[1:2]

polys_1 = []
push!(polys_1, u[1] .* u[2])
push!(polys_1, u[1]^2 .* u[2])
push!(polys_1, u[1] .* u[2]^2)
push!(polys_1, u[1]^2 .* u[2]^2)
basis_1 = Basis(polys_1, u)
res_1 = solve(problem_1, basis_1, STLSQ())
sool_1 = get_basis(res_1)
par_1 = get_parameter_map(sool_1)
print(sool_1)

# # # Beta Section
beta_result = 12995900 * NN_res'./(S .* I) 
X2 = vcat(AveT, AveH)
plot(beta_result)


# # # # SR Part
options = SymbolicRegression.Options(
    binary_operators = (+, -, *,),
    unary_operators = (exp, sin,),
    npopulations = 50,
    ncyclesperiteration = 300,
    alpha = 0.100000f0,
    # maxsize = 16,
    maxsize = 30,
)
hallOfFrame_2 = EquationSearch(X2, beta_result', niterations = 500, options = options)


# # Optimization Part
function loss_2(u)
    pred = u[1] .+ u[2] .* AveT .+ u[3] .* AveH
    # pred = exp.(u[1] .* AveH .+ u[2]) .+ u[3]
    # pred = u[1] .+ u[2] .* sin.(u[3] .* new_tspan)
    # pred = u[1] .* (1 .+ (u[2] .* AveT .+ u[3] .* AveH) .* sin.(u[4] .* new_tspan))
    sum(abs2, (beta_result .- pred))
end
# u0 = [0.3 0.02 -0.16]
# u0 = [0.3 0.02 -0.16]
u0 = [0.3 0.02 -0.16]
# u0 = [0.3 0.02 -0.16 0.01]
adtype = Optimization.AutoZygote()
optf_cases_22 = Optimization.OptimizationFunction((x, p) -> loss_2(x), adtype)
optprob_cases122 = Optimization.OptimizationProblem(optf_cases_22, u0)
sol22 = solve(optprob_cases122, ADAM(0.0005), maxiters = 10000)


# # SINDy Part
X3 = vcat(X2, new_tspan)
y = beta_result'
problem_2 = DirectDataDrivenProblem(X3, y)
@variables u[1:3]

polys_2 = polynomial_basis(u, 1)
push!(polys_2, exp.(0.033946 .* u[2] .+ -0.245741))
push!(polys_2, sin.(-0.158193 .* u[3]))
basis_2 = Basis(polys_2, u)
res_2 = solve(problem_2, basis_2, STLSQ())
sool_2 = get_basis(res_2)
par_2 = get_parameter_map(sool_2)
print(sool_2)
