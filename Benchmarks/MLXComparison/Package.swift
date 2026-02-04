// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MagmaMLXBenchmark",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0"),
    ],
    targets: [
        .executableTarget(
            name: "MagmaMLXBenchmark",
            dependencies: [
                .product(name: "Magma", package: "Magma"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources"
        ),
    ]
)
