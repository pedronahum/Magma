// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GPT2Shakespeare",
    platforms: [.macOS(.v14)],
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
