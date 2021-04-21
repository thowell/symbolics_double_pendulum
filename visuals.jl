using Colors
using CoordinateTransformations
using GeometryBasics
using MeshCat
using MeshIO
using Rotations


function default_background!(vis)
    setvisible!(vis["/Background"], true)
    setprop!(vis["/Background"], "top_color", RGBA(1.0, 1.0, 1.0, 1.0))
    setprop!(vis["/Background"], "bottom_color", RGBA(1.0, 1.0, 1.0, 1.0))
    setvisible!(vis["/Axes"], false)
end

function visualize_double_pendulum!(vis, model, x;
      r = 0.025, Δt = 0.1)

	default_background!(vis)

	# kinematics
	function k1com(model::DoublePendulum, q)
		θ1, θ2 = q

		[0.5 * model.l1 * sin(θ1);
		 -0.5 * model.l1 * cos(θ1)]
	end

	function k1ee(model::DoublePendulum, q)
		θ1, θ2 = q

		[model.l1 * sin(θ1);
		 -model.l1 * cos(θ1)]
	end

	function k2com(model::DoublePendulum, q)
		θ1, θ2 = q

		[model.l1 * sin(θ1) + 0.5 * model.l2 * sin(θ1 + θ2);
		 -model.l1 * cos(θ1) - 0.5 * model.l2 * cos(θ1 + θ2)]
	end

	function k2ee(model::DoublePendulum, q)
		θ1, θ2 = q

		[model.l1 * sin(θ1) + model.l2 * sin(θ1 + θ2);
		 -model.l1 * cos(θ1) - model.l2 * cos(θ1 + θ2)]
	end

	link1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l1),
		convert(Float32, 0.015))
	setobject!(vis["link1"], link1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	link2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l1),
		convert(Float32, 0.015))
	setobject!(vis["link2"], link2,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	T = length(x)

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	for t = 1:T
		q = x[t][1:2]

		MeshCat.atframe(anim, t) do

			_k1ee = k1ee(model, q)
			p1ee = [_k1ee[1], 0.0, _k1ee[2]]

			_k2ee = k2ee(model, q)
			p2ee = [_k2ee[1], 0.0, _k2ee[2]]


			settransform!(vis["link1"], cable_transform([0.0; 0.0; 0.0], p1ee))
			settransform!(vis["link2"], cable_transform(p1ee, p2ee))
		end
	end

	settransform!(vis["/Cameras/default"],
	    compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(-pi / 2.0))))

	MeshCat.setanimation!(vis, anim)
end

function cable_transform(y, z)
    v1 = [0.0, 0.0, 1.0]
    v2 = y[1:3,1] - z[1:3,1]
    normalize!(v2)
    ax = cross(v1, v2)
    ang = acos(v1'*v2)
    R = AngleAxis(ang, ax...)

    if any(isnan.(R))
        R = I
    else
        nothing
    end

    compose(Translation(z), LinearMap(R))
end
