using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra, Symbolics, DifferentialEquations, JLD2

struct DoublePendulum{T}
	m1::T
	m2::T
	l1::T
	l2::T
end

n = 2 # number of states
model = DoublePendulum(1.0, 1.0, 1.0, 1.0) # model

# kinematics
function kinematics_1(model::DoublePendulum, q)
	θ1, θ2 = q

	[0.5 * model.l1 * sin(θ1);
	 -0.5 * model.l1 * cos(θ1)]
end

function kinematics_2(model::DoublePendulum, q)
	θ1, θ2 = q

	[model.l1 * sin(θ1) + 0.5 * model.l2 * sin(θ1 + θ2);
	 -model.l1 * cos(θ1) - 0.5 * model.l2 * cos(θ1 + θ2)]
end

# fast kinematics functions
@variables q[1:n]
@variables q̇[1:n]

k1 = kinematics_1(model, q)
k2 = kinematics_2(model, q)

k1_exp = Symbolics.build_function(k1, q)
k2_exp = Symbolics.build_function(k2, q)

k1_func = eval(k1_exp[1])
k2_func = eval(k2_exp[1])

# kinematics Jacobians
j1 = Symbolics.jacobian(k1, q, simplify = true)
j2 = Symbolics.jacobian(k2, q, simplify = true)

j1_exp = Symbolics.build_function(j1, q)
j2_exp = Symbolics.build_function(j2, q)

j1_func = eval(j1_exp[1])
j2_func = eval(j2_exp[1])

# Lagrangian
function lagrangian(model, q, q̇)
	L = 0.0

	# mass 1
	v1 = j1_func(q) * q̇
	L += 0.5 * model.m1 * transpose(v1) * v1 		# kinetic energy
	L -= model.m1 * 9.81 * k1_func(q)[2]            # potential energy

	# mass 2
	v2 = j2_func(q) * q̇
	L += 0.5 * model.m2 * transpose(v2) * v2
	L -= model.m2 * 9.81 * k2_func(q)[2]

	return L
end

# fast Lagrangian
L = lagrangian(model, q, q̇)

dLq = Symbolics.gradient(L, q, simplify = true)
dLq̇ = Symbolics.gradient(L, q̇, simplify = true)
ddL = Symbolics.hessian(L, [q; q̇], simplify = true)

# mass matrix
M = ddL[n .+ (1:n), n .+ (1:n)]
M = simplify.(M)

# dynamics bias
C = ddL[n .+ (1:n), 1:n] * q̇ - dLq
C = simplify.(C)

# dynamics
# ẋ = [q̇; M \ (-1.0 * C)]
ẋ = [q̇; M \ (-0.5 * q̇ -1.0 * C)]
ẋ = simplify.(ẋ)

ẋ_exp = Symbolics.build_function(ẋ, q, q̇)
dynamics = eval(ẋ_exp[1])

# save dynamics function
path = joinpath(pwd(), "dynamics.jld2")
# @save path ẋ_exp
# @load path ẋ_exp

# DifferentialEquations.jl
function dynamics!(ẋ, x, p, t)
	ẋ .= dynamics(view(x, 1:n), view(x, n .+ (1:n)))
end

# simulate
x0 = [0.5 * π; 0.0; 0.0; 0.0]
tspan = (0.0, 10.0)
dt = 0.01
prob = ODEProblem(dynamics!, x0, tspan)
sol = solve(prob, Tsit5(), adaptive = false, dt = dt)

# MeshCat.jl
include(joinpath(pwd(), "visuals.jl"))
vis = Visualizer()
render(vis)
visualize_double_pendulum!(vis, model, sol.u, Δt = dt)
