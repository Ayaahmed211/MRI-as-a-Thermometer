using KomaMRI, NPZ

obj = brain_phantom2D()      

println(" number of points : ", length(obj.x))

npzwrite("phantom/phantom_raw.npz", Dict(
    "x"  => obj.x,
    "y"  => obj.y,
    "PD" => obj.ρ,
    "T1" => obj.T1,
    "T2" => obj.T2,
    "T2s" => obj.T2s,
))
println("saved successfully")