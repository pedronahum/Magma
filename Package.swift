// swift-tools-version:5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// MARK: - Build Configuration

/// Set via environment: MAGMA_XLA_PATH=/path/to/xla
let xlaPath = Context.environment["MAGMA_XLA_PATH"] ?? "/opt/xla"

/// Set MAGMA_ENABLE_XLA=1 to enable XLA linking
/// When not set, builds without XLA (stub-only mode for development)
let enableXLA = Context.environment["MAGMA_ENABLE_XLA"] == "1"

/// Set MAGMA_ENABLE_METAL=1 to enable Metal backend via MetalHLO
/// Automatically enabled on macOS if MetalHLO is available
#if os(macOS)
let enableMetal = Context.environment["MAGMA_ENABLE_METAL"] != "0"
#else
let enableMetal = false
#endif

// Conditionally add linker settings for XLA
// Note: The PJRT plugin is loaded dynamically via dlopen, so we don't need
// to link against it at build time. We just need the -ldl flag on Linux.
var xlaRuntimeLinkerSettings: [LinkerSetting] = []
if enableXLA {
    #if os(Linux)
    xlaRuntimeLinkerSettings = [
        .linkedLibrary("dl"),  // For dlopen/dlsym
    ]
    #endif
}

// MetalHLO is only available/needed on macOS. The manifest runs on the host,
// so on Linux we omit the package dependency and its product entirely.
var packageDependencies: [Package.Dependency] = []
var xlaRuntimeDependencies: [Target.Dependency] = ["CXLARuntime"]
#if os(macOS)
packageDependencies.append(.package(path: "../MetalHLO"))
xlaRuntimeDependencies.append(
    .product(name: "MetalHLO", package: "MetalHLO", condition: .when(platforms: [.macOS]))
)
#endif

let package = Package(
    name: "Magma",
    platforms: [
        .macOS(.v14),
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

        // ════════════════════════════════════════════════════════════════════
        // EXECUTABLES - Examples
        // ════════════════════════════════════════════════════════════════════
        .executable(
            name: "MNISTExample",
            targets: ["MNISTExample"]
        ),
        .executable(
            name: "BuildingSimulation",
            targets: ["BuildingSimulation"]
        ),
        .executable(
            name: "Benchmarks",
            targets: ["Benchmarks"]
        ),
        .executable(
            name: "ValueLayersExample",
            targets: ["ValueLayersExample"]
        ),
    ],
    dependencies: packageDependencies,
    targets: [
        // ════════════════════════════════════════════════════════════════════
        // LAYER 0: C Bindings to PJRT (header-only when XLA not enabled)
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
                // Define HAS_XLA when XLA is available
                enableXLA ? .define("HAS_XLA") : nil,
                // Define HAS_METAL when Metal backend is available
                enableMetal ? .define("HAS_METAL") : nil,
            ].compactMap { $0 },
            linkerSettings: xlaRuntimeLinkerSettings
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
        // EXAMPLES
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
            dependencies: ["LazyTensor"],
            path: "Tests/LazyTensorTests"
        ),

        // XLARuntime tests - requires XLA installed
        .testTarget(
            name: "XLARuntimeTests",
            dependencies: ["XLARuntime"],
            path: "Tests/XLARuntimeTests"
        ),

        // Magma tests - end-to-end integration
        .testTarget(
            name: "MagmaTests",
            dependencies: ["Magma"],
            path: "Tests/CoreTests"
        ),
    ],
    swiftLanguageVersions: [.v5]
)

// The Metal example is macOS-only (it exercises the Metal backend).
#if os(macOS)
package.products.append(
    .executable(name: "MetalExample", targets: ["MetalExample"])
)
package.targets.append(
    .executableTarget(
        name: "MetalExample",
        dependencies: ["Magma"],
        path: "Examples/Metal"
    )
)
#endif
