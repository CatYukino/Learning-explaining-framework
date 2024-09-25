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
using CairoMakie, Makie
using Makie.Colors
rng = Random.default_rng()
Random.seed!(714)

ann_cases_1 = Lux.Chain(
    Lux.Dense(1, 64, sin_cases), 
    ss,
    Lux.Dense(64, 1)
)
ann_cases_2 = Lux.Chain(
    Lux.Dense(3, 64, sin_cases), 
    ss,
    Lux.Dense(64, 1)
)

function model_cases_1(du, u, p, t)
    Ann_cases = ann_cases([t], p, st_cases)
    du[1] = -Ann_cases[1][1] .+ 0.14 .* (12995900 - u[1] - u[2])
    du[2] = Ann_cases[1][1] - (1/3) .* u[2]
    du[3] = Ann_cases[1][1]
end

function model_cases_2(du, u, p, t)
    Ann_cases = ann_cases([t, T.(t), AH.(t)], p, st_cases)
    du[1] = -Ann_cases[1][1] .+ 0.14 .* (12995900 - u[1] - u[2])
    du[2] = Ann_cases[1][1] - (1/3) .* u[2]
    du[3] = Ann_cases[1][1]
end

prob_cases2 = ODEProblem(model_cases_1, [12995900; Cases_al[:, 1]; Cases_total[:, 1]], tspan, pfinal_cases)
snn_1 = solve(prob_cases2, Tsit5(), saveat = 1)
prob_cases3 = ODEProblem(model_cases_2, [12995900; Cases_al[:, 1]; Cases_total[:, 1]], tspan, pfinal_caseses)
snn_2 = solve(prob_cases3, Tsit5(), saveat = 1)

result_1 = []
result_2 = []
for i in 1:1:357
    push!(result_1, snn_1[3, i+1] - snn_1[3, i])
end
for i in 1:1:357
    push!(result_2, snn_2[3, i+1] - snn_2[3, i])
end
scatter(Cases_al', w = 3, label = "Daily Cases", xlabel = "Weeks", ylabel = "Numbers")
plot!(result_1, w = 3, label = "Training Result 1")
plot!(result_2, w = 3, label = "Training Result 2")

# figure 2
tbl1 = (
    x = [1, 1, 2, 2],
    height1 = [113182, 336797, 30470, 152110],
    grp = [1, 2, 1, 2]
)
result_11 = [Cases_al'[1]; convert(Array{Float64}, result_1)]
result_22 = [Cases_al'[1]; convert(Array{Float64}, result_2)]
fff = CairoMakie.Figure(resolution = (1300, 1100))
ax1 = CairoMakie.Axis(fff[1,1:2],
                title = "(A)",
                xlabel = "Time(Weeks from Jan. 1st, 2010 to Nov. 10th, 2016)",
                ylabel = "ILI cases")
xs = range(0.0, 357.0)
CairoMakie.scatter!(ax1, xs, convert(Array{Float64}, Cases_daily_new')[1:358], markersize = 10, color = "#866194", label = "ILI cases")
CairoMakie.lines!(ax1, xs, result_11, linewidth = 4, color = "#88F551", label = "Fitted by NN(t)")
CairoMakie.lines!(ax1, xs, result_22, linewidth = 4, linestyle = :dash, color = "#1CBAC4", label = "Fitted by NN(T, H, t)")
CairoMakie.xlims!(ax1, 0.0, 358.0)
CairoMakie.ylims!(ax1, -2.0, 200.0)
CairoMakie.axislegend(ax1, framevisible = false, orientation = :horizontal)


ax2 = CairoMakie.Axis(fff[2, 1],
                width = 500,
                height = 450,
                title = "(B)",
                xticks = (1:2, ["Error", "Running time(seconds)"]),
)

CairoMakie.barplot!(ax2, tbl1.x, tbl1.height1, dodge = tbl1.grp, color = tbl1.grp, colormap = ["#53ABD8", "#A4CAE5"])
labels = ["Result of NN(t)", "Result of NN(T, H, t)"]
elements = [PolyElement(polycolor = "#53ABD8"), PolyElement(polycolor = "#A4CAE5")]
Legend(fff[2, 1], elements, labels, halign = :right, valign = :top, framevisible = false)


ax3 = CairoMakie.Axis(fff[2, 2],
                title = "(C)",
                xlabel = "Time(Weeks from Jan. 1st, 2010 to Nov. 10th, 2016)")
CairoMakie.lines!(ax3, xs, NN_res', linewidth = 4, color = "#F975FF", label = "B(t) estimated by NN(t)")
CairoMakie.lines!(ax3, xs, NN_reses', linewidth = 4, linestyle = :dash, color = "#EEB8FF", label = "B(t) estimated by NN(T, H, t)")
CairoMakie.axislegend(ax3, framevisible = false, orientation = :horizontal, halign = :left, valign = :top)
CairoMakie.xlims!(ax3, 0.0, 358.0)
fff
save("Figure22632.png", fff)

# figure 1
ff = CairoMakie.Figure(resolution = (1200, 900))
ax1 = CairoMakie.Axis(ff[1, 1],
                ylabel = "ILI cases",
                title = "(A)")
CairoMakie.scatter!(ax1, xs, convert(Array{Float64}, Cases_daily_new')[1:358], markersize = 10, color = "#F685FF", label = "Original ILI cases")
CairoMakie.scatter!(ax1, xs, Cases_al', markersize = 10, color = "#33FAFF", label ="Smoothed ILI cases")
CairoMakie.xlims!(ax1, 0.0, 358.0)
CairoMakie.axislegend(ax1, framevisible = true, orientation = :horizontal, halign = :right, valign = :top)
ax2 = CairoMakie.Axis(ff[2, 1],
                ylabel = "Temperature(℃)",
                title = "(B)")
CairoMakie.scatter!(ax2, xs, T_daily_new'./10, linewidth = 4, color = "#B02EFF", label = "Original temperature")
CairoMakie.scatter!(ax2, xs, AveT', markersize = 10, color = "#FF3533", label ="Smoothed temperature")
CairoMakie.xlims!(ax2, 0.0, 358.0)
CairoMakie.ylims!(ax2, -6.0, 33.0)
CairoMakie.axislegend(ax2, framevisible = true, orientation = :horizontal, halign = :right, valign = :top)
ax3 = CairoMakie.Axis(ff[3, 1],
                xlabel = "Time(Weeks from Jan. 1st, 2010 to Nov. 10th, 2016)",
                ylabel = "Absolute Humidity(kg/kg)",
                title = "(C)"
)
CairoMakie.scatter!(ax3, xs, ah_daily_new'[1:358], linewidth = 4, color = "#6800F0", label = "Original absolute humidity")
CairoMakie.scatter!(ax3, xs, AveH', markersize = 10, color = "#29FA00", label ="Smoothed absolute humidity")
CairoMakie.xlims!(ax3, 0.0, 358.0)
CairoMakie.ylims!(ax3, 0.0, 2.0e-4)
CairoMakie.axislegend(ax3, framevisible = true, orientation = :horizontal, halign = :right, valign = :top)
ff
save("Figure11.png", ff)

# figure 3
f_SR1(x1, x2) = (((((x1 + x2) + x2) * 0.00022240995519891795) + -2890.0203660351617) * x2) 
f_SR2(x1, x2) = ((((x1 * x2) * 0.00011618465326402936) + (x2 * -1509.4588823599968)) - 8.470475794338382)
f_bili(x1, x2) = ((x2 * 2.569886391621769e-8) * x1)
f_SINDy(x1, x2) = (0.3339798655/12995900) * x1 * x2
f_stand(x1, x2) = (0.3339594721/12995900) * x1 * x2
f3 = CairoMakie.Figure(resolution = (1300, 800))
ax = CairoMakie.Axis(f3[1, 1],
                xlabel = "Time(Weeks from Jan. 1st, 2010 to Nov. 10th, 2016)",
                ylabel = "Incidence rate",)
CairoMakie.scatter!(ax, xs, NN_res', markersize = 10, color = "#866194", label = "Incidence rate inferred by NN(t)")
CairoMakie.lines!(ax, xs, f_SR1.(S, I), linewidth = 4, color = "#00F00D", label = "SR model 1")
CairoMakie.lines!(ax, xs, f_SR2.(S, I), linewidth = 4, color = "#FF0025", label = "SR model 2")
CairoMakie.lines!(ax, xs, f_SINDy.(S, I), linewidth = 4, linestyle = :dash, color = "#F400FF", label = "SINDy model 1")
CairoMakie.lines!(ax, xs, f_bili.(S, I), linewidth = 4, linestyle = :dashdot, color = "#00B1FF", label = "Bilinear incidence model")
CairoMakie.lines!(ax, xs, f_stand.(S, I), linewidth = 4, linestyle = :dashdot, color = "#FFA400", label = "Standard incidence model")
CairoMakie.xlims!(ax, 0.0, 358.0)
CairoMakie.ylims!(ax, -2.0, 150.0)
CairoMakie.axislegend(ax, framevisible = false, orientation = :horizontal, halign = :left, valign = :top)
f3
save("Figure35.png", f3)


# figure 4
f_SR1(x1, x2) =  (sin(sin(sin(sin(exp(x2) * -7825.969241702319)))) - sin(sin(sin(sin(sin(x1 * 0.015734631495462822))))))
f_SR2(x1, x2) = ((1.3125460220270098 - sin(x1 * ((x2 * 37.977567304365394) + 0.0076887420581627786))) - exp(sin(x2 * -2980.111797346204)))
f_SR3(x1, x2) = (((x1 - exp(sin((x1 + 0.708128773997562) * exp(x1 + (x2 + 0.4875249868607899))) * (sin((exp(x1 * 0.6745136915953833) * -0.3492926573353199) + 0.356137445319912) + 3.5022025159095436))) * -0.0024172280369671164) + 0.356137445319912)   
f_SINDy(x1) = 9.495556934404868e8 + 3.2233282455664072e7 * x1 + -1.2140718855853813e9 * exp(0.033946 * x1 - 0.245741)
# f_linear(x1, x2) = 0.334859 + 7.99701e-7 * x1 + -0.174505 * x2
# f_exp(x1, x2) = exp(0.033946 * x2 + -0.245741) + -0.447269
# f_sin(x1) = 0.334858 + -7.63937e-11 * sin(-0.158193 * x1)
# f_mix(x1, x2, x3) = 0.334858 * (1 + (8.1953e-7 * x1 + -0.174668 * x2) * sin(0.0086311 * x3))
f_linear(x1, x2) = 0.366219 + -0.00531414 * x1 + 634.556 * x2
f_exp1(x1, x2) = 0.366229 * exp(-0.0151317 * x1 + 1771.93 * x2)
f_exp2(x2) = 2.18834 + -7.31712e-7 * exp(0.00129993 * x2 + 14.7251)
f_sin(x1) = 0.335508 + -0.0527556 * sin(0.12 * x1)
f_mix(x1, x2, x3) = 0.350093 * (1 + (-0.0147609 * x1 + 2125.86 * x2) * sin(0.00716068 * x3))
f4 = CairoMakie.Figure(resolution = (1300, 800))
ax = CairoMakie.Axis(f4[1, 1],
                xlabel = "Time(Weeks from Jan. 1st, 2010 to Nov. 10th, 2016)",
                ylabel = "β(t)"
)
CairoMakie.scatter!(ax, xs, beta_result, markersize = 10, color = "#866194", label = "β(t) inferred from NN(t) and standard model")
CairoMakie.lines!(ax, xs, f_SR1.(AveT, AveH)'[1:358], linewidth = 4, color = "#00FFE8", label = "SR model 3")
CairoMakie.lines!(ax, xs, f_SR2.(AveT, AveH)'[1:358], linewidth = 4, color = "#FFA100", label = "SR model 4")
CairoMakie.lines!(ax, xs, f_SR3.(AveT, AveH)'[1:358], linewidth = 3, color = "#00F00D", label = "SR model 5")
CairoMakie.lines!(ax, xs, f_SINDy.(AveH)'[1:358], linewidth = 4, linestyle = :dash, color = "#FF2C00", label = "SINDy model 2")
CairoMakie.lines!(ax, xs, f_linear.(AveT, AveH)'[1:358], linewidth = 4, linestyle = :dashdot, color = "#5BDCB8", label = "Linear model β1")
CairoMakie.lines!(ax, xs, f_exp1.(AveT, AveH)'[1:358], linewidth = 4, linestyle = :dashdot, color = "#FF00ED", label = "Exponential model 1 β2")
CairoMakie.lines!(ax, xs, f_exp2.(AveH)'[1:358], linewidth = 4, linestyle = :dashdot, color = "#B22222", label = "Exponential model 2 β3")
CairoMakie.lines!(ax, xs, f_sin.(new_tspan)'[1:358], linewidth = 4, linestyle = :dashdot, color = "#AE6637", label = "Sinusoid model β4")
CairoMakie.lines!(ax, xs, f_mix.(AveT, AveH, new_tspan)'[1:358], linewidth = 4, linestyle = :dashdot, color = "#0028FF", label = "Mixed model β5")
CairoMakie.xlims!(ax, 0.0, 358.0)
CairoMakie.ylims!(ax, 0.13, 0.7)
CairoMakie.axislegend(ax, framevisible = false, halign = :right, valign = :top)
f4
save("Figure4445.png", f4)
