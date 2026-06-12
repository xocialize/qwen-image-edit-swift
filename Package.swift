// swift-tools-version: 6.2
// qwen-image-edit-swift — Swift/MLX mirror of Qwen-Image-Edit-2511 (Apache-2.0):
// Qwen2.5-VL-7B-conditioned 20B double-stream DiT (zero_cond_t) + 3D causal VAE.
// Reference = diffusers 0.37.1 QwenImageEditPlusPipeline (NOT mflux — see the
// deviation ledger in /Volumes/DEV_ARCHIVE/qwen-image-edit-mlx/PORTING-SPEC.md).
// Phases gate on the P2 goldens at VideoResearch/qwen-image-edit-models/goldens.

import PackageDescription

let package = Package(
    name: "QwenImageEdit",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "QwenImageEdit", targets: ["QwenImageEdit"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "QwenImageEdit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/QwenImageEdit"
        ),
        .testTarget(
            name: "QwenImageEditTests",
            dependencies: ["QwenImageEdit"],
            path: "Tests/QwenImageEditTests"
        ),
    ]
)
