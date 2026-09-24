// swift-tools-version:6.0
// Magma requires a Swift 6.0+ toolchain from swift.org (the `_Differentiation`
// module used for autodiff is not shipped with Xcode's toolchain).

import PackageDescription

// MARK: - Build Configuration
//
// The PJRT plugin (CPU/GPU/TPU) is loaded at runtime with dlopen from
// $MAGMA_XLA_PATH, so no XLA libraries are needed at build time and the
// default build is fully functional once a plugin is installed.

/// Metal backend via MetalHLO (macOS only, opt-in).
///
/// Set MAGMA_ENABLE_METAL=1 to add the MetalHLO dependency. By default it comes
/// from GitHub; set MAGMA_METALHLO_PATH to use a local checkout instead. It is
/// opt-in so that the default package graph has no unversioned dependencies and
/// Magma can be consumed as a tagged release.
#if os(macOS)
let enableMetal = Context.environment["MAGMA_ENABLE_METAL"] == "1"
#else
let enableMetal = false
#endif

var packageDependencies: [Package.Dependency] = []
var xlaRuntimeDependencies: [Target.Dependency] = ["CXLARuntime"]
if enableMetal {
    if let metalHLOPath = Context.environment["MAGMA_METALHLO_PATH"] {
        packageDependencies.append(.package(path: metalHLOPath))
    } else {
        packageDependencies.append(
            .package(url: "https://github.com/pedronahum/MetalHLO.git", branch: "main"))
    }
    xlaRuntimeDependencies.append(
        .product(name: "MetalHLO", package: "MetalHLO", condition: .when(platforms: [.macOS]))
    )
    // The Metal bridge converts between StableHLO and MetalHLO element types.
    xlaRuntimeDependencies.append("StableHLO")
}

let package = Package(
    name: "Magma",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        // ════════════════════════════════════════════════════════════════════
        // MAIN PRODUCT - What users import
        // ════════════════════════════════════════════════════════════════════
        .library(
            name: "Magma",
            targets: ["Magma"]
        ),

        // ════════════════════════════════════════════════════════════════════
        // INDIVIDUAL LAYERS - For advanced users
        // ════════════════════════════════════════════════════════════════════
        .library(
            name: "LazyTensor",
            targets: ["LazyTensor"]
        ),
        .library(
            name: "StableHLO",
            targets: ["StableHLO"]
        ),
        .library(
            name: "XLARuntime",
            targets: ["XLARuntime"]
        ),
    ],
    dependencies: packageDependencies,
    targets: [
        // ════════════════════════════════════════════════════════════════════
        // LAYER 0: C Bindings to PJRT (plugin loaded at runtime via dlopen)
        // ════════════════════════════════════════════════════════════════════
        .target(
            name: "CXLARuntime",
            path: "Sources/CXLARuntime",
            publicHeadersPath: "include"
        ),

        // ════════════════════════════════════════════════════════════════════
        // LAYER 1: XLA Runtime (Swift wrapper around PJRT + MetalHLO on macOS)
        // ════════════════════════════════════════════════════════════════════
        .target(
            name: "XLARuntime",
            dependencies: xlaRuntimeDependencies,
            path: "Sources/XLARuntime",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ],
            // dlopen/dlsym live in libdl on glibc < 2.34.
            linkerSettings: [.linkedLibrary("dl", .when(platforms: [.linux]))]
        ),

        // ════════════════════════════════════════════════════════════════════
        // LAYER 2: StableHLO (Pure Swift MLIR generation)
        // ════════════════════════════════════════════════════════════════════
        .target(
            name: "StableHLO",
            dependencies: [],  // PURE SWIFT - No dependencies!
            path: "Sources/StableHLO",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),

        // ════════════════════════════════════════════════════════════════════
        // LAYER 3: Lazy Tensor (x10-style execution)
        // ════════════════════════════════════════════════════════════════════
        .target(
            name: "LazyTensor",
            dependencies: ["StableHLO", "XLARuntime"],
            path: "Sources/LazyTensor",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),

        // ════════════════════════════════════════════════════════════════════
        // LAYER 4: Magma (Main tensor API - import Magma)
        // ════════════════════════════════════════════════════════════════════
        .target(
            name: "Magma",
            dependencies: ["LazyTensor"],
            path: "Sources/Core",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),

        // ════════════════════════════════════════════════════════════════════
        // EXAMPLES - run with `swift run <TargetName>` (not exported as products)
        // ════════════════════════════════════════════════════════════════════
        .executableTarget(
            name: "MNISTExample",
            dependencies: ["Magma"],
            path: "Examples/MNIST"
        ),
        .executableTarget(
            name: "BuildingSimulation",
            dependencies: ["Magma"],
            path: "Examples/BuildingSimulation"
        ),
        .executableTarget(
            name: "Benchmarks",
            dependencies: ["Magma"],
            path: "Examples/Benchmarks"
        ),
        .executableTarget(
            name: "ValueLayersExample",
            dependencies: ["Magma"],
            path: "Examples/ValueLayers"
        ),
        // ════════════════════════════════════════════════════════════════════
        // TESTS
        // ════════════════════════════════════════════════════════════════════

        // StableHLO tests - NO XLA REQUIRED (pure Swift)
        .testTarget(
            name: "StableHLOTests",
            dependencies: ["StableHLO"],
            path: "Tests/StableHLOTests"
        ),

        // LazyTensor tests - with mocking
        .testTarget(
            name: "LazyTensorTests",
            dependencies: ["LazyTensor", "StableHLO"],
            path: "Tests/LazyTensorTests"
        ),

        // XLARuntime tests - requires XLA installed. Some suites (XLA integration,
        // Metal) drive the full stack, so they depend on the upper layers too.
        .testTarget(
            name: "XLARuntimeTests",
            dependencies: ["XLARuntime", "StableHLO", "LazyTensor", "Magma"],
            path: "Tests/XLARuntimeTests"
        ),

        // Magma tests - end-to-end integration
        .testTarget(
            name: "MagmaTests",
            dependencies: ["Magma"],
            path: "Tests/CoreTests"
        ),
    ],
    swiftLanguageModes: [.v5]
)

// The Metal example needs the (opt-in) MetalHLO backend.
if enableMetal {
    package.targets.append(
        .executableTarget(
            name: "MetalExample",
            dependencies: ["Magma"],
            path: "Examples/Metal"
        )
    )
}
