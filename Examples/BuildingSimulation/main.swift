// BuildingSimulation - Magma Port
// Ported from PyTorch building simulation benchmark
// Original: https://github.com/PassiveLogic/differentiable-swift-examples
//
// This example demonstrates:
// - Differentiable simulation with Tensor operations
// - Computing gradients through a multi-timestep simulation
// - Using indexing operations for structured data access
//
// Usage:
//   ./BuildingSimulation [magma|native|both] [trials] [timesteps]
//   Default: both 1000 1000

import Foundation
import _Differentiation
import Magma
import LazyTensor

// MARK: - Configuration

var benchmarkTrials = 1000
var benchmarkTimesteps = 1000
let warmup = 3
let dTime: Float = 0.1
let printGradToCompare = false

let pi: Float = 3.14159265359

// MARK: - Type Definitions (as tensor indices)

// TubeType indices
enum TubeTypeIndices {
    static let tubeSpacing = 0
    static let diameter = 1
    static let thickness = 2
    static let resistivity = 3
}

// SlabType indices
enum SlabTypeIndices {
    static let temp = 0
    static let area = 1
    static let cp = 2
    static let density = 3
    static let thickness = 4
}

// QuantaType indices
enum QuantaIndices {
    static let power = 0
    static let temp = 1
    static let flow = 2
    static let density = 3
    static let cp = 4
}

// TankType indices
enum TankTypeIndices {
    static let temp = 0
    static let volume = 1
    static let cp = 2
    static let density = 3
    static let mass = 4
}

// SimParams indices
enum SimParamsIndices {
    static let tube = 0
    static let slab = 1
    static let quanta = 2
    static let tank = 3
    static let startingTemp = 4
}

// MARK: - Initial Data

// TubeType: [tubeSpacing, diameter, thickness, resistivity, padding]
let TubeTypeTensor = Tensor<Float>([0.50292, 0.019, 0.001588, 2.43, 0.0], shape: [5])

// SlabType: [temp, area, Cp, density, thickness]
let SlabTypeTensor = Tensor<Float>([21.1111111, 100.0, 0.2, 2242.58, 0.101], shape: [5])

// QuantaType: [power, temp, flow, density, Cp]
let QuantaTypeTensor = Tensor<Float>([0.0, 60.0, 0.0006309, 1000.0, 4180.0], shape: [5])

// TankType: [temp, volume, Cp, density, mass]
let TankTypeTensor = Tensor<Float>([70.0, 0.0757082, 4180.0, 1000.0, 75.708], shape: [5])

// Starting temperature (padded to match other tensor sizes)
let startingTemperature = Tensor<Float>([33.3, 0, 0, 0, 0], shape: [5])

// MARK: - Helper Masks

// Mask for zeroing first element: [0, 1, 1, 1, 1]
let maskZeroFirst = Tensor<Float>([0.0, 1.0, 1.0, 1.0, 1.0], shape: [5])

// Mask for selecting first element: [1, 0, 0, 0, 0]
let maskSelectFirst = Tensor<Float>([1.0, 0.0, 0.0, 0.0, 0.0], shape: [5])

// MARK: - Computation Functions

/// Compute thermal resistance between floor and tubing
@differentiable(reverse)
func computeResistance(
    floor: Tensor<Float>,
    tube: Tensor<Float>,
    quanta: Tensor<Float>
) -> Tensor<Float> {
    let geometryCoeff: Float = 10.0

    // tubingSurfaceArea = (floor.area / tube.tubeSpacing) * pi * tube.diameter
    let floorArea = floor[SlabTypeIndices.area]
    let tubeSpacing = tube[TubeTypeIndices.tubeSpacing]
    let tubeDiameter = tube[TubeTypeIndices.diameter]

    let tubingSurfaceArea = (floorArea / tubeSpacing) * Tensor<Float>.full([], pi) * tubeDiameter

    // resistance_abs = tube.resistivity * tube.thickness / tubingSurfaceArea
    let tubeResistivity = tube[TubeTypeIndices.resistivity]
    let tubeThickness = tube[TubeTypeIndices.thickness]

    let resistanceAbs = tubeResistivity * tubeThickness / tubingSurfaceArea

    // Apply geometry correction
    let resistanceCorrected = resistanceAbs * Tensor<Float>.full([], geometryCoeff)

    return resistanceCorrected
}

/// Compute power transferred to the building load
@differentiable(reverse)
func computeLoadPower(
    floor: Tensor<Float>,
    tube: Tensor<Float>,
    quanta: Tensor<Float>
) -> (quanta: Tensor<Float>, loadPower: Tensor<Float>) {
    let resistanceAbs = computeResistance(floor: floor, tube: tube, quanta: quanta)

    // conductance = 1 / resistance
    let conductance = Tensor<Float>.ones([]) / resistanceAbs

    // dTemp = floor.temp - quanta.temp
    let floorTemp = floor[SlabTypeIndices.temp]
    let quantaTemp = quanta[QuantaIndices.temp]
    let dTemp = floorTemp - quantaTemp

    // power = dTemp * conductance
    let power = dTemp * conductance

    // loadPower = -power (negative because heat flows out of floor)
    let loadPower = -power

    // Update quanta power: quanta * [0,1,1,1,1] + power * [1,0,0,0,0]
    let resultQuanta = quanta * maskZeroFirst + power.reshape([1]).broadcast(to: [5]) * maskSelectFirst

    return (resultQuanta, loadPower)
}

/// Update quanta temperature based on power and flow
@differentiable(reverse)
func updateQuanta(quanta: Tensor<Float>) -> Tensor<Float> {
    // workingVolume = quanta.flow * dTime
    let quantaFlow = quanta[QuantaIndices.flow]
    let workingVolume = quantaFlow * Tensor<Float>.full([], dTime)

    // workingMass = workingVolume * quanta.density
    let quantaDensity = quanta[QuantaIndices.density]
    let workingMass = workingVolume * quantaDensity

    // workingEnergy = quanta.power * dTime
    let quantaPower = quanta[QuantaIndices.power]
    let workingEnergy = quantaPower * Tensor<Float>.full([], dTime)

    // TempRise = workingEnergy / quanta.Cp / workingMass
    let quantaCp = quanta[QuantaIndices.cp]
    let tempRise = workingEnergy / quantaCp / workingMass

    // Update temp: quanta + TempRise * [0,1,0,0,0]
    let tempMask = Tensor<Float>([0.0, 1.0, 0.0, 0.0, 0.0], shape: [5])
    var resultQuanta = quanta + tempRise.reshape([1]).broadcast(to: [5]) * tempMask

    // Zero out power: result * [0,1,1,1,1]
    resultQuanta = resultQuanta * maskZeroFirst

    return resultQuanta
}

/// Update building model (floor) temperature
@differentiable(reverse)
func updateBuildingModel(power: Tensor<Float>, floor: Tensor<Float>) -> Tensor<Float> {
    // floorVolume = floor.area * floor.thickness
    let floorArea = floor[SlabTypeIndices.area]
    let floorThickness = floor[SlabTypeIndices.thickness]
    let floorVolume = floorArea * floorThickness

    // floorMass = floorVolume * floor.density
    let floorDensity = floor[SlabTypeIndices.density]
    let floorMass = floorVolume * floorDensity

    // floorTempChange = (power * dTime) / floor.Cp / floorMass
    let floorCp = floor[SlabTypeIndices.cp]
    let floorTempChange = (power * Tensor<Float>.full([], dTime)) / floorCp / floorMass

    // Update temp: floor + floorTempChange * [1,0,0,0,0]
    let resultFloor = floor + floorTempChange.reshape([1]).broadcast(to: [5]) * maskSelectFirst

    return resultFloor
}

/// Update source tank and quanta based on heat exchange
@differentiable(reverse)
func updateSourceTank(
    store: Tensor<Float>,
    quanta: Tensor<Float>
) -> (store: Tensor<Float>, quanta: Tensor<Float>) {
    // massPerTime = quanta.flow * quanta.density
    let quantaFlow = quanta[QuantaIndices.flow]
    let quantaDensity = quanta[QuantaIndices.density]
    let massPerTime = quantaFlow * quantaDensity

    // dTemp = store.temp - quanta.temp
    let storeTemp = store[TankTypeIndices.temp]
    let quantaTemp = quanta[QuantaIndices.temp]
    let dTemp = storeTemp - quantaTemp

    // power = dTemp * massPerTime * quanta.Cp
    let quantaCp = quanta[QuantaIndices.cp]
    let power = dTemp * massPerTime * quantaCp

    // Update quanta power
    let updatedQuanta = quanta * maskZeroFirst + power.reshape([1]).broadcast(to: [5]) * maskSelectFirst

    // tankMass = store.volume * store.density
    let storeVolume = store[TankTypeIndices.volume]
    let storeDensity = store[TankTypeIndices.density]
    let tankMass = storeVolume * storeDensity

    // TempRise = (power * dTime) / store.Cp / tankMass
    let storeCp = store[TankTypeIndices.cp]
    let tempRise = (power * Tensor<Float>.full([], dTime)) / storeCp / tankMass

    // Update store temp
    let updatedStore = store + tempRise.reshape([1]).broadcast(to: [5]) * maskSelectFirst

    return (updatedStore, updatedQuanta)
}

// Ground truth target for loss calculation
let groundTruthTarget = Tensor<Float>.full([], 27.344767)

/// Loss function: absolute difference from target
@differentiable(reverse)
func lossCalc(pred: Tensor<Float>) -> Tensor<Float> {
    return (pred - groundTruthTarget).abs()
}

// MARK: - Simulation

/// Run the full building simulation
@differentiable(reverse)
func simulate(simParams: Tensor<Float>, timesteps: Int) -> Tensor<Float> {
    // Extract components from simParams (2D tensor: [5, 5])
    let pexTube = simParams[SimParamsIndices.tube]
    var slab = simParams[SimParamsIndices.slab]
    var tank = simParams[SimParamsIndices.tank]
    var quanta = simParams[SimParamsIndices.quanta]

    // Set initial slab temperature from startingTemp
    let startTemp = simParams[SimParamsIndices.startingTemp][0]
    slab = slab * maskZeroFirst + startTemp.reshape([1]).broadcast(to: [5]) * maskSelectFirst

    // Run simulation for timesteps iterations
    for _ in 0..<timesteps {
        // Update source tank
        let tankResult = updateSourceTank(store: tank, quanta: quanta)
        tank = tankResult.store
        quanta = tankResult.quanta

        // Update quanta after tank exchange
        quanta = updateQuanta(quanta: quanta)

        // Compute load power and update quanta
        let loadResult = computeLoadPower(floor: slab, tube: pexTube, quanta: quanta)
        quanta = loadResult.quanta
        let powerToBuilding = loadResult.loadPower

        // Update quanta after load
        quanta = updateQuanta(quanta: quanta)

        // Update building model
        slab = updateBuildingModel(power: powerToBuilding, floor: slab)
    }

    // Return final slab temperature
    return slab[SlabTypeIndices.temp]
}

/// Full pipeline: simulate and compute loss
@differentiable(reverse)
func fullPipe(simParams: Tensor<Float>, timesteps: Int) -> Tensor<Float> {
    let pred = simulate(simParams: simParams, timesteps: timesteps)
    let loss = lossCalc(pred: pred)
    return loss
}

// MARK: - Benchmarking

func measureTime<T>(_ function: () -> T) -> (time: Double, result: T) {
    let start = DispatchTime.now()
    let result = function()
    let end = DispatchTime.now()
    let elapsed = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1E9
    return (elapsed, result)
}

func runMagmaBenchmark(trials: Int, timesteps: Int) {
    print()
    print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
    print("Magma Building Simulation Benchmark")
    print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
    print("Trials: \(trials)")
    print("Timesteps: \(timesteps)")
    print("Warmup iterations: \(warmup)")
    print()

    // Stack initial parameters into a 2D tensor [5, 5]
    let SimParamsConstant = Tensor<Float>.stack([
        TubeTypeTensor,
        SlabTypeTensor,
        QuantaTypeTensor,
        TankTypeTensor,
        startingTemperature
    ], axis: 0)

    print("SimParams shape: \(SimParamsConstant.shape)")

    var totalForwardTime: Double = 0
    var totalGradientTime: Double = 0

    for i in 0..<(trials + warmup) {
        // Forward pass
        let (forwardTime, forwardOutput) = measureTime {
            fullPipe(simParams: SimParamsConstant, timesteps: timesteps)
        }

        // Trigger materialization
        LazyTensorBarrier()
        let lossValue = forwardOutput.item()

        // Gradient computation
        let (gradientTime, gradientResult) = measureTime {
            gradient(at: SimParamsConstant) { params in
                fullPipe(simParams: params, timesteps: timesteps)
            }
        }

        // Trigger materialization of gradients
        LazyTensorBarrier()

        if printGradToCompare {
            print("Gradient shape: \(gradientResult.shape)")
            print("Gradient values: \(gradientResult.scalars().prefix(10))...")
        }

        if i >= warmup {
            totalForwardTime += forwardTime
            totalGradientTime += gradientTime

            if i == warmup {
                print("Loss value: \(lossValue)")
            }
        }
    }

    let averageForwardTime = totalForwardTime / Double(trials)
    let averageGradientTime = totalGradientTime / Double(trials)

    print()
    print("Results:")
    print("--------")
    print("Average forward only time: \(String(format: "%.6f", averageForwardTime)) seconds")
    print("Average forward + backward (gradient) time: \(String(format: "%.6f", averageGradientTime)) seconds")
}

// MARK: - Barrier Per Iteration Variant (Option B)
//
// This variant forces materialization after each iteration, leveraging XLA's compilation cache.
//
// Trade-offs vs Loop Unrolling:
// - Pro: Compilation happens once for the iteration kernel, subsequent iterations hit cache
// - Pro: Memory-efficient (don't need to hold entire unrolled graph)
// - Pro: Better for heavy per-iteration work (batch training, convolutions)
// - Con: Per-iteration kernel launch overhead (~3-5ms per barrier on CPU)
// - Con: XLA can't optimize across iteration boundaries
//
// Benchmark findings (100 timesteps):
// - Native Swift: 0.003s (gradient)
// - Magma Unrolled: 0.152s (gradient) - 49x slower
// - Magma Barrier: 0.508s (forward only) - kernel launch overhead dominates
//
// Conclusion: For lightweight per-iteration work, unrolling is faster.
// Barrier approach is better when per-iteration work is heavy (e.g., batch training).

/// Single iteration step - designed for barrier-per-iteration execution
func simulateOneStep(
    pexTube: Tensor<Float>,
    slab: Tensor<Float>,
    tank: Tensor<Float>,
    quanta: Tensor<Float>
) -> (slab: Tensor<Float>, tank: Tensor<Float>, quanta: Tensor<Float>) {
    var currentSlab = slab
    var currentTank = tank
    var currentQuanta = quanta

    // Update source tank
    let tankResult = updateSourceTank(store: currentTank, quanta: currentQuanta)
    currentTank = tankResult.store
    currentQuanta = tankResult.quanta

    // Update quanta after tank exchange
    currentQuanta = updateQuanta(quanta: currentQuanta)

    // Compute load power and update quanta
    let loadResult = computeLoadPower(floor: currentSlab, tube: pexTube, quanta: currentQuanta)
    currentQuanta = loadResult.quanta
    let powerToBuilding = loadResult.loadPower

    // Update quanta after load
    currentQuanta = updateQuanta(quanta: currentQuanta)

    // Update building model
    currentSlab = updateBuildingModel(power: powerToBuilding, floor: currentSlab)

    return (currentSlab, currentTank, currentQuanta)
}

/// Run simulation with barrier after each iteration
/// This allows XLA to cache the compiled iteration kernel
func simulateWithBarriers(simParams: Tensor<Float>, timesteps: Int) -> Tensor<Float> {
    // Extract components from simParams (2D tensor: [5, 5])
    let pexTube = simParams[SimParamsIndices.tube]
    var slab = simParams[SimParamsIndices.slab]
    var tank = simParams[SimParamsIndices.tank]
    var quanta = simParams[SimParamsIndices.quanta]

    // Set initial slab temperature from startingTemp
    let startTemp = simParams[SimParamsIndices.startingTemp][0]
    slab = slab * maskZeroFirst + startTemp.reshape([1]).broadcast(to: [5]) * maskSelectFirst

    // Mark initial state for materialization and execute
    slab.markForMaterialization()
    tank.markForMaterialization()
    quanta.markForMaterialization()
    pexTube.markForMaterialization()
    LazyTensorBarrier()

    // Run simulation with barrier per iteration
    for _ in 0..<timesteps {
        let result = simulateOneStep(pexTube: pexTube, slab: slab, tank: tank, quanta: quanta)
        slab = result.slab
        tank = result.tank
        quanta = result.quanta

        // Mark outputs for materialization - this is required for barrier to compile them
        slab.markForMaterialization()
        tank.markForMaterialization()
        quanta.markForMaterialization()

        // Barrier: compile once, execute from cache on subsequent iterations
        LazyTensorBarrier()
    }

    // Return final slab temperature
    return slab[SlabTypeIndices.temp]
}

/// Full pipeline with barriers
func fullPipeWithBarriers(simParams: Tensor<Float>, timesteps: Int) -> Tensor<Float> {
    let pred = simulateWithBarriers(simParams: simParams, timesteps: timesteps)
    let loss = lossCalc(pred: pred)
    LazyTensorBarrier()
    return loss
}

func runMagmaBarrierBenchmark(trials: Int, timesteps: Int) {
    print()
    print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
    print("Magma Building Simulation (BARRIER PER ITERATION)")
    print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
    print("Trials: \(trials)")
    print("Timesteps: \(timesteps)")
    print("Warmup iterations: \(warmup)")
    print()

    // Stack initial parameters into a 2D tensor [5, 5]
    let SimParamsConstant = Tensor<Float>.stack([
        TubeTypeTensor,
        SlabTypeTensor,
        QuantaTypeTensor,
        TankTypeTensor,
        startingTemperature
    ], axis: 0)

    print("SimParams shape: \(SimParamsConstant.shape)")

    var totalForwardTime: Double = 0

    // Track cache stats for this benchmark
    let cache = CompilationCache.shared
    let initialHits = cache.hitCount
    let initialMisses = cache.missCount

    for i in 0..<(trials + warmup) {
        // Forward pass with barriers
        let (forwardTime, forwardOutput) = measureTime {
            fullPipeWithBarriers(simParams: SimParamsConstant, timesteps: timesteps)
        }

        let lossValue = forwardOutput.item()

        if i >= warmup {
            totalForwardTime += forwardTime

            if i == warmup {
                print("Loss value: \(lossValue)")
            }
        }
    }

    let averageForwardTime = totalForwardTime / Double(trials)

    // Calculate benchmark-specific cache stats
    let benchmarkHits = cache.hitCount - initialHits
    let benchmarkMisses = cache.missCount - initialMisses
    let hitRate = Double(benchmarkHits) / Double(max(1, benchmarkHits + benchmarkMisses)) * 100

    print()
    print("Results:")
    print("--------")
    print("Average forward time: \(String(format: "%.6f", averageForwardTime)) seconds")
    print("Cache hit rate: \(String(format: "%.1f", hitRate))% (\(benchmarkHits) hits, \(benchmarkMisses) misses)")
    print("Note: ~\(timesteps + 2) barriers per trial, each ~3-5ms overhead on CPU")
}

// MARK: - Scan-Based Simulation (Option A - with full autodiff)
//
// This variant uses the `scan` primitive which:
// - Traces the body function once
// - Stores intermediate states for reverse-mode autodiff
// - Can be optimized to XLA while_loop in the future
//
// Unlike the barrier approach, scan supports full gradient computation.

/// Building simulation state for scan
struct SimState: Differentiable, ScanState {
    var slab: Tensor<Float>
    var tank: Tensor<Float>
    var quanta: Tensor<Float>
    @noDerivative var pexTube: Tensor<Float>

    init(slab: Tensor<Float>, tank: Tensor<Float>, quanta: Tensor<Float>, pexTube: Tensor<Float>) {
        self.slab = slab
        self.tank = tank
        self.quanta = quanta
        self.pexTube = pexTube
    }

    // ScanState conformance
    func toTensors() -> [Tensor<Float>] {
        [slab, tank, quanta]
    }

    static func fromTensors(_ tensors: [Tensor<Float>]) -> SimState {
        // pexTube is a constant, not carried through the loop
        fatalError("SimState.fromTensors not supported - pexTube must be provided")
    }

    static var tensorCount: Int { 3 }
}

/// Single iteration step for scan
@differentiable(reverse)
func simulationStep(state: SimState) -> SimState {
    var currentSlab = state.slab
    var currentTank = state.tank
    var currentQuanta = state.quanta

    // Update source tank
    let tankResult = updateSourceTank(store: currentTank, quanta: currentQuanta)
    currentTank = tankResult.store
    currentQuanta = tankResult.quanta

    // Update quanta after tank exchange
    currentQuanta = updateQuanta(quanta: currentQuanta)

    // Compute load power and update quanta
    let loadResult = computeLoadPower(floor: currentSlab, tube: state.pexTube, quanta: currentQuanta)
    currentQuanta = loadResult.quanta
    let powerToBuilding = loadResult.loadPower

    // Update quanta after load
    currentQuanta = updateQuanta(quanta: currentQuanta)

    // Update building model
    currentSlab = updateBuildingModel(power: powerToBuilding, floor: currentSlab)

    return SimState(slab: currentSlab, tank: currentTank, quanta: currentQuanta, pexTube: state.pexTube)
}

/// Run simulation using scan primitive
@differentiable(reverse)
func simulateWithScan(simParams: Tensor<Float>, timesteps: Int) -> Tensor<Float> {
    // Extract components from simParams (2D tensor: [5, 5])
    let pexTube = simParams[SimParamsIndices.tube]
    var slab = simParams[SimParamsIndices.slab]
    let tank = simParams[SimParamsIndices.tank]
    let quanta = simParams[SimParamsIndices.quanta]

    // Set initial slab temperature from startingTemp
    let startTemp = simParams[SimParamsIndices.startingTemp][0]
    slab = slab * maskZeroFirst + startTemp.reshape([1]).broadcast(to: [5]) * maskSelectFirst

    // Create initial state
    let initialState = SimState(slab: slab, tank: tank, quanta: quanta, pexTube: pexTube)

    // Run simulation using scan
    let finalState = scan(iterations: timesteps, initialState: initialState, body: simulationStep)

    // Return final slab temperature
    return finalState.slab[SlabTypeIndices.temp]
}

/// Full pipeline with scan
@differentiable(reverse)
func fullPipeWithScan(simParams: Tensor<Float>, timesteps: Int) -> Tensor<Float> {
    let pred = simulateWithScan(simParams: simParams, timesteps: timesteps)
    let loss = lossCalc(pred: pred)
    return loss
}

func runMagmaScanBenchmark(trials: Int, timesteps: Int) {
    print()
    print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
    print("Magma Building Simulation (SCAN with VJP)")
    print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
    print("Trials: \(trials)")
    print("Timesteps: \(timesteps)")
    print("Warmup iterations: \(warmup)")
    print()

    // Stack initial parameters into a 2D tensor [5, 5]
    let SimParamsConstant = Tensor<Float>.stack([
        TubeTypeTensor,
        SlabTypeTensor,
        QuantaTypeTensor,
        TankTypeTensor,
        startingTemperature
    ], axis: 0)

    print("SimParams shape: \(SimParamsConstant.shape)")
    print("Strategy: scan() with full reverse-mode autodiff")
    print()

    var totalForwardTime: Double = 0
    var totalGradientTime: Double = 0

    for i in 0..<(trials + warmup) {
        // Forward pass
        let (forwardTime, forwardOutput) = measureTime {
            fullPipeWithScan(simParams: SimParamsConstant, timesteps: timesteps)
        }

        // Trigger materialization
        LazyTensorBarrier()
        let lossValue = forwardOutput.item()

        // Gradient computation
        let (gradientTime, gradientResult) = measureTime {
            gradient(at: SimParamsConstant) { params in
                fullPipeWithScan(simParams: params, timesteps: timesteps)
            }
        }

        // Trigger materialization of gradients
        LazyTensorBarrier()

        if i >= warmup {
            totalForwardTime += forwardTime
            totalGradientTime += gradientTime

            if i == warmup {
                print("Loss value: \(lossValue)")
                if printGradToCompare {
                    print("Gradient shape: \(gradientResult.shape)")
                    print("Gradient values: \(gradientResult.scalars().prefix(10))...")
                }
            }
        }
    }

    let averageForwardTime = totalForwardTime / Double(trials)
    let averageGradientTime = totalGradientTime / Double(trials)

    print()
    print("Results:")
    print("--------")
    print("Average forward only time: \(String(format: "%.6f", averageForwardTime)) seconds")
    print("Average forward + backward (gradient) time: \(String(format: "%.6f", averageGradientTime)) seconds")
}

// MARK: - Main

// Parse command line arguments
var mode = "both"
if CommandLine.arguments.count > 1 {
    mode = CommandLine.arguments[1].lowercased()
}
if CommandLine.arguments.count > 2 {
    benchmarkTrials = Int(CommandLine.arguments[2]) ?? 1000
}
if CommandLine.arguments.count > 3 {
    benchmarkTimesteps = Int(CommandLine.arguments[3]) ?? 1000
}

print("Building Simulation Benchmark")
print("=============================")
print("Mode: \(mode)")
print("Trials: \(benchmarkTrials)")
print("Timesteps: \(benchmarkTimesteps)")
print()
print("Available modes:")
print("  native      - Native Swift (differentiable structs)")
print("  magma - Magma with loop unrolling")
print("  barrier     - Magma with barrier per iteration")
print("  scan        - Magma with scan primitive (full autodiff)")
print("  both        - Native + Magma (unrolled)")
print("  all         - All four variants")

if mode == "native" || mode == "both" || mode == "all" {
    runNativeSwiftBenchmark(trials: benchmarkTrials, timesteps: benchmarkTimesteps)
}

if mode == "magma" || mode == "both" || mode == "all" {
    runMagmaBenchmark(trials: benchmarkTrials, timesteps: benchmarkTimesteps)
}

if mode == "barrier" || mode == "all" {
    runMagmaBarrierBenchmark(trials: benchmarkTrials, timesteps: benchmarkTimesteps)
}

if mode == "scan" || mode == "all" {
    runMagmaScanBenchmark(trials: benchmarkTrials, timesteps: benchmarkTimesteps)
}

if mode == "both" || mode == "all" {
    print()
    print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
    print("Comparison Complete")
    print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
}
