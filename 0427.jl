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
# plot(NN_res', label = "Training Result", w = 3, xlabel = "Weeks", ylabel = "Numbers")
# # # # # savefig("NN_t.png")
# @save "./20240816_1.bson" ann_cases
# psave_1 = collect(pfinal_cases)
# @save "./20240816_1_1.bson" psave_1

ir(t) = ann_cases([t], pfinal_cases, st_cases)[1][1]
plot(ir.(new_tspan)')
function RecoverDE(du, u, p, t)
    du[1] = -ir.(t) + 0.14 * (12995900 - u[1] - u[2])
    du[2] = ir.(t) - (1/3) * u[2]
    du[3] = ir.(t)
end
prob_re = ODEProblem(RecoverDE, [12995900; Cases_al[:, 1]; Cases_total[:, 1]], tspan)
sol_re = solve(prob_re, saveat = 1, sensealg = ForwardDiffSensitivity())

# AIC part
# function aic_cal(k, res)
#     return 2 * k + 179 * log(res / 358)
# end

# function aicc(k, res)
#     return aic_cal(k ,res) + (2 * k ^ 2 + 2 * k)/(357 - k)
# end

# res1 = sum(abs2, (NN_res' .- f_SR1.(S, I)))
# res2 = sum(abs2, (NN_res' .- f_SR2.(S, I)))
# res3 = sum(abs2, (NN_res' .- f_SINDy.(S, I)))
# res4 = sum(abs2, (NN_res' .- f_bili.(S, I)))

# aic_SR1 = aic_cal(3, res1)
# aic_SR2 = aic_cal(3, res2)
# aic_SINDy = aic_cal(1, res3)
# aic_Tra = aic_cal(1, res4)

# aicc_SR1 = aicc(3, res1)
# aicc_SR2 = aicc(3, res2)
# aicc_SINDy = aicc(1, res3)
# aicc_Tra = aicc(1, res4)



# # # SymbolicRegression Part
S = sol_re[1, :]
I = sol_re[2, :]
# X1 = convert(Array{Float64}, vcat(S'./12995900, I'))
# options = SymbolicRegression.Options(
#     # binary_operators = (+, -, *,),
#     binary_operators = (*,),
#     npopulations = 50,
#     ncyclesperiteration = 300,
#     alpha = 0.100000f0,
#     maxsize = 10,
# )
# hallOfFrame_1 = EquationSearch(X1, NN_res, niterations = 100, options = options)
# f_SR1(x1, x2) = (((((x1 + x2) + x2) * 0.00022240995519891795) + -2890.0203660351617) * x2) 
# f_SR2(x1, x2) = ((((x1 * x2) * 0.00011618465326402936) + (x2 * -1509.4588823599968)) - 8.470475794338382)
# f_bili(x1, x2) = ((x2 * 2.569886391621769e-8) * x1)


# # # # SINDy Part
# # # y = NN_res .* 12995900
# # # problem_1 = DirectDataDrivenProblem(X1, y)
# # # @variables u[1:2]

# # # polys_1 = []
# # # push!(polys_1, u[1] .* u[2])
# # # push!(polys_1, u[1]^2 .* u[2])
# # # push!(polys_1, u[1] .* u[2]^2)
# # # push!(polys_1, u[1]^2 .* u[2]^2)
# # # basis_1 = Basis(polys_1, u)
# # # res_1 = solve(problem_1, basis_1, STLSQ())
# # # sool_1 = get_basis(res_1)
# # # par_1 = get_parameter_map(sool_1)
# # # print(sool_1)
# f_SINDy(x1, x2) = (0.3339798655/12995900) * x1 * x2
# mae1 = log(sum(abs, (NN_res' .- f_SR1.(S, I))) / 358)
# mae2 = log(sum(abs, (NN_res' .- f_SR2.(S, I))) / 358)
# mae3 = log(sum(abs, (NN_res' .- f_SINDy.(S, I))) / 358)
# mae4 = log(sum(abs, (NN_res' .- f_bili.(S, I))) / 358)
# # # # Picture Part
# scatter(NN_res', w = 3, xlabel = "weeks", ylabel = "Numbers", label = "Training Result")
# plot!(f_SR1.(S, I), label = "SR Result 1", w = 3)
# plot!(f_SR2.(S, I), label = "SR Result 2", w = 3)
# plot!(f_SINDy.(S, I), label = "SINDy Result", w = 3)
# plot!(f_bili.(S, I), label = "Bilinear", w = 3)
# # # # savefig("NN_t result1.png")
# m14 = -(mae1 - mae4)/(11 - 5)
# m24 = -(mae2 - mae4)/(11 - 5)


# # # # Loss Part
# sum(abs2, (NN_res' .- f_SR1.(S, I)))
# sum(abs2, (NN_res' .- f_SR2.(S, I)))
# sum(abs2, (NN_res' .- f_SINDy.(S, I)))
# sum(abs2, (NN_res' .- f_bili.(S, I)))

# log(sum(abs2, (NN_res' .- f_SR1.(S, I))) ) + 5.0
# log(sum(abs2, (NN_res' .- f_SR2.(S, I))) )  + 5.0
# log(sum(abs2, (NN_res' .- f_SINDy.(S, I))) )  + 2.0
# log(sum(abs2, (NN_res' .- f_bili.(S, I))) )  + 2.0

# entropy_approx(f_SR1.(S, I); m = 2, τ = 1, r = 0.2 * Statistics.std(f_SR1.(S, I)), base = MathConstants.e)
# entropy_approx(f_SR2.(S, I); m = 2, τ = 1, r = 0.2 * Statistics.std(f_SR2.(S, I)), base = MathConstants.e)
# entropy_approx(f_SINDy.(S, I); m = 2, τ = 1, r = 0.2 * Statistics.std(f_SINDy.(S, I)), base = MathConstants.e)
# entropy_approx(f_bili.(S, I); m = 2, τ = 1, r = 0.2 * Statistics.std(f_bili.(S, I)), base = MathConstants.e)

# entropy_sample(f_SR1.(S, I); r = 0.2 * std(f_SR1.(S, I)), m = 2, τ = 1, normalize = true)
# entropy_sample(f_SR2.(S, I); r = 0.2 * std(f_SR2.(S, I)), m = 2, τ = 1, normalize = true)
# entropy_sample(f_SINDy.(S, I); r = 0.2 * std(f_SINDy.(S, I)), m = 2, τ = 1, normalize = true)
# entropy_sample(f_bili.(S, I); r = 0.2 * std(f_bili.(S, I)), m = 2, τ = 1, normalize = true)

# # # # Beta Section
# beta_result = 12995900 * NN_res'./(S .* I) 
# X2 = vcat(AveT, AveH)
# plot(beta_result)


# # # # SR Part
# options = SymbolicRegression.Options(
#     binary_operators = (+, -, *,),
#     unary_operators = (exp, sin,),
#     npopulations = 50,
#     ncyclesperiteration = 300,
#     alpha = 0.100000f0,
#     # maxsize = 16,
#     maxsize = 30,
# )
# hallOfFrame_2 = EquationSearch(X2, beta_result', niterations = 500, options = options)
# # # # f_SR1(x1, x2) =  (0.3459771388133769 - ((x1 * 0.0023492166550534185) - (exp(sin((x1 * (x1 + 1.1906745573978)) + 0.3086382867422177)) * 0.019902316483567245)))
# f_SR1(x1, x2) =  (sin(sin(sin(sin(exp(x2) * -7825.969241702319)))) - sin(sin(sin(sin(sin(x1 * 0.015734631495462822))))))
# f_SR2(x1, x2) = ((1.3125460220270098 - sin(x1 * ((x2 * 37.977567304365394) + 0.0076887420581627786))) - exp(sin(x2 * -2980.111797346204)))
# f_SR3(x1, x2) = (((x1 - exp(sin((x1 + 0.708128773997562) * exp(x1 + (x2 + 0.4875249868607899))) * (sin((exp(x1 * 0.6745136915953833) * -0.3492926573353199) + 0.356137445319912) + 3.5022025159095436))) * -0.0024172280369671164) + 0.356137445319912)  
# plot(beta_result)
# plot!(f_SR1.(AveT, AveH)')


# # # # Optimization Part
# # function loss_2(u)
# #     pred = u[1] .+ u[2] .* AveT .+ u[3] .* AveH
# #     # pred = exp.(u[1] .* AveH .+ u[2]) .+ u[3]
# #     # pred = u[1] .+ u[2] .* sin.(u[3] .* new_tspan)
# #     # pred = u[1] .* (1 .+ (u[2] .* AveT .+ u[3] .* AveH) .* sin.(u[4] .* new_tspan))
# #     sum(abs2, (beta_result .- pred))
# # end
# # # u0 = [0.3 0.02 -0.16]
# # # u0 = [0.3 0.02 -0.16]
# # u0 = [0.3 0.02 -0.16]
# # # u0 = [0.3 0.02 -0.16 0.01]
# # adtype = Optimization.AutoZygote()
# # optf_cases_22 = Optimization.OptimizationFunction((x, p) -> loss_2(x), adtype)
# # optprob_cases122 = Optimization.OptimizationProblem(optf_cases_22, u0)
# # sol22 = solve(optprob_cases122, ADAM(0.0005), maxiters = 10000)
# f_linear(x1, x2) = 0.334859 + 7.99701e-7 * x1 + -0.174505 * x2
# f_exp(x1, x2) = exp(0.033946 * x2 + -0.245741) + -0.447269
# f_sin(x1) = 0.334858 + -7.63937e-11 * sin(-0.158193 * x1)
# f_mix(x1, x2, x3) = 0.334858 * (1 + (8.1953e-7 * x1 + -0.174668 * x2) * sin(0.0086311 * x3))
# plot(beta_result)
# plot!(f_linear.(AveT, AveH)')
# plot!(f_exp.(AveT, AveH)')
# plot!(f_sin.(new_tspan)')
# plot!(f_mix.(AveT, AveH, new_tspan)')


# # # # SINDy Part
# # X3 = vcat(X2, new_tspan)
# # y = beta_result'
# # problem_2 = DirectDataDrivenProblem(X3, y)
# # @variables u[1:3]

# # polys_2 = polynomial_basis(u, 1)
# # push!(polys_2, exp.(0.033946 .* u[2] .+ -0.245741))
# # push!(polys_2, sin.(-0.158193 .* u[3]))
# # basis_2 = Basis(polys_2, u)
# # res_2 = solve(problem_2, basis_2, STLSQ())
# # sool_2 = get_basis(res_2)
# # par_2 = get_parameter_map(sool_2)
# # print(sool_2)
# f_SINDy(x1) = 9.495556934404868e8 + 3.2233282455664072e7 * x1 + -1.2140718855853813e9 * exp(0.033946 * x1 - 0.245741)
# # plot!(f_SINDy.(AveH)')

# # # # picture Part
# scatter(beta_result, w = 3, xlabel = "weeks", ylabel = "Numbers", label = "Training Result")
# plot!(f_SR1.(AveT, AveH)', label = "SR Result 1", w = 3)
# plot!(f_SR2.(AveT, AveH)', label = "SR Result 2", w = 3)
# plot!(f_SR3.(AveT, AveH)', label = "SR Result 3", w = 3)
# plot!(f_SINDy.(AveH)', label = "SINDy Result", w = 3)
# plot!(f_linear.(AveT, AveH)', label = "Linear", w = 3)
# plot!(f_exp.(AveT, AveH)', label = "Exponential", w = 3)
# plot!(f_sin.(new_tspan)', label = "Sinuoid", w = 3)
# plot!(f_mix.(AveT, AveH, new_tspan)', label = "Mixed", w = 3)
# # savefig("beta result_t.png")

# mae1 = log(sum(abs, (beta_result' .- f_SR1.(AveT, AveH))) / 358)
# mae2 = log(sum(abs, (beta_result' .- f_SR2.(AveT, AveH))) / 358)
# mae3 = log(sum(abs, (beta_result' .- f_SR3.(AveT, AveH))) / 358)
# mae4 = log(sum(abs, (beta_result' .- f_SINDy.(AveH))) / 358)
# mae5 = log(sum(abs, (beta_result' .- f_linear.(AveT, AveH))) / 358)
# mae6 = log(sum(abs, (beta_result' .- f_exp.(AveT, AveH))) / 358)
# mae7 = log(sum(abs, (beta_result' .- f_sin.(new_tspan))) / 358)
# mae8 = log(sum(abs, (beta_result' .- f_mix.(AveT, AveH, new_tspan))) / 358)

# m15 = -(mae1 - mae5)/(17  - 9)
# m25 = -(mae2 - mae5)/(16  - 9)
# m35 = -(mae3 - mae5)/(30  - 9)
# m45 = -(mae4 - mae5)/(14  - 9)



# # # # loss part
# aa1 = sum(abs2, (beta_result' .- f_SR1.(AveT, AveH)))
# aa2 = sum(abs2, (beta_result' .- f_SR2.(AveT, AveH)))
# aa3 = sum(abs2, (beta_result' .- f_SR3.(AveT, AveH)))
# aa4 = sum(abs2, (beta_result' .- f_SINDy.(AveH)))
# aa5 = sum(abs2, (beta_result' .- f_linear.(AveT, AveH)))
# aa6 = sum(abs2, (beta_result' .- f_exp.(AveT, AveH)))
# aa7 = sum(abs2, (beta_result' .- f_sin.(new_tspan)))
# aa8 = sum(abs2, (beta_result' .- f_mix.(AveT, AveH, new_tspan)))
# aic_SR1 = aic_cal(2, aa1)
# aic_SR2 = aic_cal(3, aa2)
# aic_SR3 = aic_cal(3, aa3)
# aic_SINDy = aic_cal(3, aa4)
# aic_linear = aic_cal(3, aa5)
# aic_exp = aic_cal(2, aa6)
# aic_sin = aic_cal(2, aa7)
# aic_mix = aic_cal(3, aa8)

# aicc_SR1 = aicc(2, aa1)
# aicc_SR2 = aicc(3, aa2)
# aicc_SR3 = aicc(3, aa3)
# aicc_SINDy = aicc(3, aa4)
# aicc_linear = aicc(3, aa5)
# aicc_exp = aicc(2, aa6)
# aicc_sin = aicc(2, aa7)
# aicc_mix = aicc(3, aa8)




# entropy_approx(f_SR1.(AveT, AveH)'[1:358]; m = 2, τ = 1, r = 0.2 * Statistics.std(f_SR1.(AveT, AveH)'[1:358]), base = MathConstants.e)
# entropy_approx(f_SR2.(AveT, AveH)'[1:358]; m = 2, τ = 1, r = 0.2 * Statistics.std(f_SR2.(AveT, AveH)'[1:358]), base = MathConstants.e)
# entropy_approx(f_SR3.(AveT, AveH)'[1:358]; m = 2, τ = 1, r = 0.2 * Statistics.std(f_SR3.(AveT, AveH)'[1:358]), base = MathConstants.e)
# entropy_approx(f_SINDy.(AveH)'[1:358]; m = 2, τ = 1, r = 0.2 * Statistics.std(f_SINDy.(AveH)'[1:358]), base = MathConstants.e)
# entropy_approx(f_linear.(AveT, AveH)'[1:358]; m = 2, τ = 1, r = 0.2 * Statistics.std(f_linear.(AveT, AveH)'[1:358]), base = MathConstants.e)
# entropy_approx(f_exp.(AveT, AveH)'[1:358]; m = 2, τ = 1, r = 0.2 * Statistics.std(f_exp.(AveT, AveH)'[1:358]), base = MathConstants.e)
# entropy_approx(f_sin.(new_tspan)'[1:358]; m = 2, τ = 1, r = 0.2 * Statistics.std(f_sin.(new_tspan)'[1:358]), base = MathConstants.e)
# entropy_approx(f_mix.(AveT, AveH, new_tspan)'[1:358]; m = 2, τ = 1, r = 0.2 * Statistics.std(f_mix.(AveT, AveH, new_tspan)'[1:358]), base = MathConstants.e)

# entropy_sample(f_SR1.(AveT, AveH)'[1:358]; r = 0.2 * std(f_SR1.(AveT, AveH)'[1:358]), m = 2, τ = 1, normalize = true)
# entropy_sample(f_SR2.(AveT, AveH)'[1:358]; r = 0.2 * std(f_SR2.(AveT, AveH)'[1:358]), m = 2, τ = 1, normalize = true)
# entropy_sample(f_SR3.(AveT, AveH)'[1:358]; r = 0.2 * std(f_SR3.(AveT, AveH)'[1:358]), m = 2, τ = 1, normalize = true)
# entropy_sample(f_SINDy.(AveH)'[1:358]; r = 0.2 * std(f_SINDy.(AveH)'[1:358]), m = 2, τ = 1, normalize = true)
# entropy_sample(f_linear.(AveT, AveH)'[1:358]; r = 0.2 * std(f_linear.(AveT, AveH)'[1:358]), m = 2, τ = 1, normalize = true)
# entropy_sample(f_exp.(AveT, AveH)'[1:358]; r = 0.2 * std(f_exp.(AveT, AveH)'[1:358]), m = 2, τ = 1, normalize = true)
# entropy_sample(f_sin.(new_tspan)'[1:358]; r = 0.2 * std(f_sin.(new_tspan)'[1:358]), m = 2, τ = 1, normalize = true)
# entropy_sample(f_mix.(AveT, AveH, new_tspan)'[1:358]; r = 0.2 * std(f_mix.(AveT, AveH, new_tspan)'[1:358]), m = 2, τ = 1, normalize = true)