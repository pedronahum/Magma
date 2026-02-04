// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MagmaBenchmark",
    platforms: [.macOS(.v14)],
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
