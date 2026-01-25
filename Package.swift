// swift-tools-version:5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// MARK: - Build Configuration

/// Set via environment: MAGMA_XLA_PATH=/path/to/xla
let xlaPath = Context.environment["MAGMA_XLA_PATH"] ?? "/opt/xla"

/// Set MAGMA_ENABLE_XLA=1 to enable XLA linking
/// When not set, builds without XLA (stub-only mode for development)
let enableXLA = Context.environment["MAGMA_ENABLE_XLA"] == "1"

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
    ],
    dependencies: [
        // No external Swift package dependencies!
        // XLA is linked via system library when MAGMA_ENABLE_XLA=1
    ],
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
        // LAYER 1: XLA Runtime (Swift wrapper around PJRT)
        // ════════════════════════════════════════════════════════════════════
        .target(
            name: "XLARuntime",
            dependencies: ["CXLARuntime"],
            path: "Sources/XLARuntime",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                // Define HAS_XLA when XLA is available
                enableXLA ? .define("HAS_XLA") : nil,
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
