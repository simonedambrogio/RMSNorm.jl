using Test
using Lux, Random, Statistics, NNlib
include("RMSNorm.jl")

@testset "RMSNorm Tests" begin
    # Test 1: Basic initialization
    rng = Random.default_rng()
    shape = 64
    l = RMSNorm(shape)
    ps, st = Lux.setup(rng, l)
    

    @test haskey(ps, :scale)
    @test size(ps.scale) == (64, 1)
    @test all(ps.scale .≈ 1.0f0)  # Check if scale is initialized to ones

    # Test 2: Forward pass with random input
    batch_size = 32
    x = randn(Float32, shape, batch_size)
    y, st = l(x, ps, st)
    # println(y)
    
    # Check output shape
    @test size(y) == size(x)
    
    # Test 3: Check RMS normalization property
    
    rms_values = sqrt.(mean(abs2.(y), dims=1))  # Calculate RMS over feature dimension
    
    # Debug prints
    println("Input shape: ", size(x))
    println("Output shape: ", size(y))
    println("RMS values shape: ", size(rms_values))
    println("First few RMS values: ", rms_values[1:min(5, end)])
    println("Mean RMS: ", mean(rms_values))
    println("Min RMS: ", minimum(rms_values))
    println("Max RMS: ", maximum(rms_values))
    
    @test all(isapprox.(rms_values, 1.0f0, atol=1e-5))

    # Test 4: Test with affine=false
    layer_no_affine = RMSNorm(shape, affine=false)
    ps_no_affine, st_no_affine = Lux.setup(rng, layer_no_affine)
    @test isempty(ps_no_affine)

    # Test 5: Test with different activation function
    layer_relu = RMSNorm(shape, relu)
    y_relu, _ = layer_relu(x, ps, st)
    @test all(y_relu .>= 0)  # ReLU should make all values non-negative

    # Test 6: Test with 2D input
    shape_2d = (32, 32)
    layer_2d = RMSNorm(shape_2d)
    ps_2d, st_2d = Lux.setup(rng, layer_2d)
    x_2d = randn(Float32, shape_2d..., batch_size)
    y_2d, _ = layer_2d(x_2d, ps_2d, st_2d)
    @test size(y_2d) == size(x_2d)
end
