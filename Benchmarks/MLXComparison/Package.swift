// swift-tools-version:5.9
// MLX vs Magma Metal Comparison Benchmark

import PackageDescription

let package = Package(
    name: "MLXComparison",
    platforms: [
        .macOS(.v14),
    ],
    dependencies: [
        // MLX Swift
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        // Magma (local)
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "MLXComparison",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Magma", package: "Magma"),
            ],
            path: "Sources"
        ),
    ]
)
