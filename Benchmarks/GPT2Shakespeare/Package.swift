// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "GPT2Shakespeare",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "GPT2Shakespeare",
            dependencies: [
                .product(name: "Magma", package: "Magma"),
            ],
            path: "Sources"
        ),
    ]
)
