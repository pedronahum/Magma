// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "MagmaBenchmark",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "MagmaBenchmark",
            dependencies: [
                .product(name: "Magma", package: "Magma"),
            ],
            path: "Sources"
        ),
    ]
)
