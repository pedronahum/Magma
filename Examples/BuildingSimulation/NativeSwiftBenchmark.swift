// NativeSwiftBenchmark.swift
// Native Swift implementation of the building simulation for comparison
// From: https://github.com/PassiveLogic/differentiable-swift-examples

import _Differentiation
import Foundation

// Simulation parameters - will be set from command line or defaults
var nativeTrials = 1000
var nativeTimesteps = 1000
let nativeDTime: Float = 0.1
let nativePrintGradToCompare = false

// Definitions
let nativeπ = Float.pi

struct SimParams: Differentiable {
    var tube: TubeType = .init()
    var slab: SlabType = .init()
    var quanta: QuantaType = .init()
    var tank: TankType = .init()
    var startingTemp: Float
}

struct TubeType: Differentiable {
    var tubeSpacing: Float = 0.50292 // meters
    var diameter: Float = 0.019 // m (3/4")
    var thickness: Float = 0.001588 // m (1/16")
    var resistivity: Float = 2.43 // (K/W)m
}

struct SlabType: Differentiable {
    var temp: Float = 21.1111111 // °C
    var area: Float = 100.0 // m^2
    var Cp: Float = 0.2
    var density: Float = 2242.58 // kg/m^3
    var thickness: Float = 0.101 // m
}

struct QuantaType: Differentiable {
    var power: Float = 0.0 // Watt
    var temp: Float = 60.0 // °C
    var flow: Float = 0.0006309 // m^3/sec
    var density: Float = 1000.0 // kg/m^3
    var Cp: Float = 4180.0 // ws/(kg • K)
}

struct TankType: Differentiable {
    var temp: Float = 70.0
    var volume: Float = 0.0757082
    var Cp: Float = 4180.000
    var density: Float = 1000.000
    var mass: Float = 75.708
}

// Computations

@differentiable(reverse)
func nativeComputeResistance(floor: SlabType, tube: TubeType, quanta _: QuantaType) -> Float {
    let geometry_coeff: Float = 10.0

    let tubingSurfaceArea = (floor.area / tube.tubeSpacing) * nativeπ * tube.diameter
    let resistance_abs = tube.resistivity * tube.thickness / tubingSurfaceArea

    let resistance_corrected = resistance_abs * geometry_coeff

    return resistance_corrected
}

struct QuantaAndPower: Differentiable {
    var quanta: QuantaType
    var power: Float
}

@differentiable(reverse)
func nativeComputeLoadPower(floor: SlabType, tube: TubeType, quanta: QuantaType) -> QuantaAndPower {
    let resistance_abs = nativeComputeResistance(floor: floor, tube: tube, quanta: quanta)

    let conductance: Float = 1 / resistance_abs
    let dTemp = floor.temp - quanta.temp
    let power = dTemp * conductance

    var updatedQuanta = quanta
    updatedQuanta.power = power
    let loadPower = -power

    return QuantaAndPower(quanta: updatedQuanta, power: loadPower)
}

@differentiable(reverse)
func nativeUpdateQuanta(quanta: QuantaType) -> QuantaType {
    let workingVolume = (quanta.flow * nativeDTime)
    let workingMass = (workingVolume * quanta.density)
    let workingEnergy = quanta.power * nativeDTime
    let TempRise = workingEnergy / quanta.Cp / workingMass
    var updatedQuanta = quanta
    updatedQuanta.temp = quanta.temp + TempRise

    updatedQuanta.power = 0
    return updatedQuanta
}

@differentiable(reverse)
func nativeUpdateBuildingModel(power: Float, floor: SlabType) -> SlabType {
    var updatedFloor = floor

    let floorVolume = floor.area * floor.thickness
    let floorMass = floorVolume * floor.density

    updatedFloor.temp = floor.temp + ((power * nativeDTime) / floor.Cp / floorMass)
    return updatedFloor
}

struct TankAndQuanta: Differentiable {
    var tank: TankType
    var quanta: QuantaType
}

@differentiable(reverse)
func nativeUpdateSourceTank(store: TankType, quanta: QuantaType) -> TankAndQuanta {
    var updatedStore = store
    var updatedQuanta = quanta

    let massPerTime = quanta.flow * quanta.density
    let dTemp = store.temp - quanta.temp
    let power = dTemp * massPerTime * quanta.Cp

    updatedQuanta.power = power

    let tankMass = store.volume * store.density
    let TempRise = (power * nativeDTime) / store.Cp / tankMass
    updatedStore.temp = store.temp + TempRise

    return TankAndQuanta(tank: updatedStore, quanta: updatedQuanta)
}

@differentiable(reverse)
@inlinable public func absDifferentiable(_ value: Float) -> Float {
    if value < 0 {
        return -value
    }
    return value
}

func nativeLossCalc(pred: Float, gt: Float) -> Float {
    let diff = pred - gt
    return absDifferentiable(diff)
}

// Simulations

@differentiable(reverse)
func nativeSimulate(simParams: SimParams) -> Float {
    let pexTube = simParams.tube
    var slab = simParams.slab
    var tank = simParams.tank
    var quanta = simParams.quanta

    slab.temp = simParams.startingTemp
    for _ in 0 ..< nativeTimesteps {
        let tankAndQuanta = nativeUpdateSourceTank(store: tank, quanta: quanta)
        tank = tankAndQuanta.tank
        quanta = tankAndQuanta.quanta

        quanta = nativeUpdateQuanta(quanta: quanta)

        let quantaAndPower = nativeComputeLoadPower(floor: slab, tube: pexTube, quanta: quanta)
        quanta = quantaAndPower.quanta
        let powerToBuilding = quantaAndPower.power
        quanta = nativeUpdateQuanta(quanta: quanta)

        slab = nativeUpdateBuildingModel(power: powerToBuilding, floor: slab)
    }
    return slab.temp
}

var blackHole: Any?
@inline(never)
func dontLetTheCompilerOptimizeThisAway<T>(_ x: T) {
    blackHole = x
}

func nativeMeasure<T>(_ block: () throws -> T) throws -> (time: Double, result: T) {
    let t0 = DispatchTime.now()
    let result = try block()
    let t1 = DispatchTime.now()
    let elapsed = Double(t1.uptimeNanoseconds - t0.uptimeNanoseconds) / 1E9
    return (elapsed, result)
}

@differentiable(reverse)
func nativeFullPipe(simParams: SimParams) -> Float {
    let pred = nativeSimulate(simParams: simParams)
    let loss = nativeLossCalc(pred: pred, gt: 27.344767)
    return loss
}

func runNativeSwiftBenchmark(trials: Int, timesteps: Int) {
    nativeTrials = trials
    nativeTimesteps = timesteps

    var simParams = SimParams(startingTemp: 33.3)

    var totalPureForwardTime: Double = 0
    var totalGradientTime: Double = 0
    let warmup = 3

    print("Native Swift Building Simulation Benchmark")
    print("==========================================")
    print("Trials: \(nativeTrials)")
    print("Timesteps: \(nativeTimesteps)")
    print("Warmup: \(warmup)")
    print()

    for i in 0 ..< (nativeTrials + warmup) {
        let (forwardOnly, forwardResult) = try! nativeMeasure {
            return nativeFullPipe(simParams: simParams)
        }
        dontLetTheCompilerOptimizeThisAway(forwardResult)

        let (gradientTime, grad) = try! nativeMeasure {
            return gradient(at: simParams, of: nativeFullPipe)
        }
        dontLetTheCompilerOptimizeThisAway(grad)

        if i >= warmup {
            totalPureForwardTime += forwardOnly
            totalGradientTime += gradientTime

            if i == warmup {
                print("Loss value: \(forwardResult)")
            }
        }
    }

    let averagePureForward = totalPureForwardTime / Double(nativeTrials)
    let averageGradient = totalGradientTime / Double(nativeTrials)

    print()
    print("Results:")
    print("--------")
    print("Average forward only time: \(String(format: "%.6f", averagePureForward)) seconds")
    print("Average forward + backward (gradient) time: \(String(format: "%.6f", averageGradient)) seconds")
}
