// Magma - Transform Tests
// Tests for PyTorch-compatible data transforms

import XCTest
@testable import Magma
@testable import LazyTensor

// MARK: - Compose Tests

final class ComposeTransformTests: XCTestCase {

    func testComposeEmptyList() {
        let compose = transforms.Compose([])
        let input = Tensor<Float>.ones([2, 4, 4, 3])
        let output = compose(input)
        XCTAssertEqual(output.shape, input.shape)
    }

    func testComposeSingleTransform() {
        let compose = transforms.Compose([
            transforms.Identity()
        ])
        let input = Tensor<Float>.ones([2, 4, 4, 3])
        let output = compose(input)
        XCTAssertEqual(output.shape, input.shape)
    }

    func testComposeMultipleTransforms() {
        let compose = transforms.Compose([
            transforms.Normalize(mean: [0.5], std: [0.5]),
            transforms.Identity()
        ])
        let input = Tensor<Float>.ones([2, 28, 28, 1])
        let output = compose(input)
        XCTAssertEqual(output.shape, [2, 28, 28, 1])
    }
}

// MARK: - Normalize Tests

final class NormalizeTransformTests: XCTestCase {

    func testNormalizeSingleChannel() {
        let normalize = transforms.Normalize(mean: [0.5], std: [0.5])
        let input = Tensor<Float>.ones([2, 28, 28, 1])
        let output = normalize(input)

        XCTAssertEqual(output.shape, [2, 28, 28, 1])

        // (1.0 - 0.5) / 0.5 = 1.0
        let values = output.scalars()
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-5)
    }

    func testNormalizeMultiChannel() {
        let normalize = transforms.Normalize(
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225]
        )
        let input = Tensor<Float>.ones([1, 4, 4, 3])
        let output = normalize(input)

        XCTAssertEqual(output.shape, [1, 4, 4, 3])
    }

    func testNormalize2D() {
        // For [batch, features] input
        let normalize = transforms.Normalize(mean: [0.0], std: [1.0])
        let input = Tensor<Float>.randn([32, 10])
        let output = normalize(input)

        XCTAssertEqual(output.shape, [32, 10])
    }

    func testDenormalize() {
        let mean: [Float] = [0.5, 0.5, 0.5]
        let std: [Float] = [0.25, 0.25, 0.25]

        let normalize = transforms.Normalize(mean: mean, std: std)
        let denormalize = transforms.Denormalize(mean: mean, std: std)

        let input = Tensor<Float>.randn([1, 8, 8, 3])
        let normalized = normalize(input)
        let recovered = denormalize(normalized)

        XCTAssertEqual(recovered.shape, input.shape)
    }
}

// MARK: - RandomApply Tests

final class RandomApplyTransformTests: XCTestCase {

    func testRandomApplyAlways() {
        // p=1.0 should always apply
        let apply = transforms.RandomApply([
            transforms.Normalize(mean: [0.5], std: [0.5])
        ], p: 1.0)

        let input = Tensor<Float>.ones([2, 8, 8, 1])
        let output = apply(input)

        XCTAssertEqual(output.shape, input.shape)
    }

    func testRandomApplyNever() {
        // p=0.0 should never apply
        let apply = transforms.RandomApply([
            transforms.Normalize(mean: [0.5], std: [0.5])
        ], p: 0.0)

        let input = Tensor<Float>.ones([2, 8, 8, 1])
        let output = apply(input)

        XCTAssertEqual(output.shape, input.shape)
        // Should be unchanged
        let values = output.scalars()
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-5)
    }
}

// MARK: - RandomChoice Tests

final class RandomChoiceTransformTests: XCTestCase {

    func testRandomChoiceSingleOption() {
        let choice = transforms.RandomChoice([
            transforms.Identity()
        ])

        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = choice(input)

        XCTAssertEqual(output.shape, input.shape)
    }

    func testRandomChoiceMultipleOptions() {
        let choice = transforms.RandomChoice([
            transforms.Identity(),
            transforms.Normalize(mean: [0.5], std: [0.5])
        ])

        let input = Tensor<Float>.ones([2, 8, 8, 1])
        let output = choice(input)

        XCTAssertEqual(output.shape, input.shape)
    }
}

// MARK: - CenterCrop Tests

final class CenterCropTransformTests: XCTestCase {

    func testCenterCrop4D() {
        let crop = transforms.CenterCrop(size: (4, 4))
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = crop(input)

        XCTAssertEqual(output.shape, [2, 4, 4, 3])
    }

    func testCenterCrop3D() {
        let crop = transforms.CenterCrop(size: (6, 6))
        let input = Tensor<Float>.ones([10, 10, 3])
        let output = crop(input)

        XCTAssertEqual(output.shape, [6, 6, 3])
    }

    func testCenterCropSquare() {
        let crop = transforms.CenterCrop(size: 14)
        let input = Tensor<Float>.ones([1, 28, 28, 1])
        let output = crop(input)

        XCTAssertEqual(output.shape, [1, 14, 14, 1])
    }

    func testCenterCropSameSize() {
        let crop = transforms.CenterCrop(size: (8, 8))
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = crop(input)

        XCTAssertEqual(output.shape, [1, 8, 8, 3])
    }
}

// MARK: - RandomCrop Tests

final class RandomCropTransformTests: XCTestCase {

    func testRandomCrop4D() {
        let crop = transforms.RandomCrop(size: (4, 4))
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = crop(input)

        XCTAssertEqual(output.shape, [2, 4, 4, 3])
    }

    func testRandomCrop3D() {
        let crop = transforms.RandomCrop(size: (6, 6))
        let input = Tensor<Float>.ones([10, 10, 3])
        let output = crop(input)

        XCTAssertEqual(output.shape, [6, 6, 3])
    }

    func testRandomCropSquare() {
        let crop = transforms.RandomCrop(size: 14)
        let input = Tensor<Float>.ones([1, 28, 28, 1])
        let output = crop(input)

        XCTAssertEqual(output.shape, [1, 14, 14, 1])
    }
}

// MARK: - Pad Tests

final class PadTransformTests: XCTestCase {

    func testPadUniform() {
        let pad = transforms.Pad(padding: 2)
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = pad(input)

        XCTAssertEqual(output.shape, [1, 12, 12, 3])
    }

    func testPadNonUniform() {
        let pad = transforms.Pad(padding: (1, 2, 3, 4))  // top, bottom, left, right
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = pad(input)

        XCTAssertEqual(output.shape, [1, 11, 15, 3])  // 8+1+2=11, 8+3+4=15
    }

    func testPad3D() {
        let pad = transforms.Pad(padding: 1)
        let input = Tensor<Float>.ones([8, 8, 3])
        let output = pad(input)

        XCTAssertEqual(output.shape, [10, 10, 3])
    }
}

// MARK: - Flip Tests

final class FlipTransformTests: XCTestCase {

    func testRandomHorizontalFlipAlways() {
        // Not easily testable due to randomness, but check shape
        let flip = transforms.RandomHorizontalFlip(p: 1.0)
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = flip(input)

        XCTAssertEqual(output.shape, [2, 8, 8, 3])
    }

    func testRandomHorizontalFlipNever() {
        let flip = transforms.RandomHorizontalFlip(p: 0.0)
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = flip(input)

        XCTAssertEqual(output.shape, [2, 8, 8, 3])
    }

    func testRandomVerticalFlipAlways() {
        let flip = transforms.RandomVerticalFlip(p: 1.0)
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = flip(input)

        XCTAssertEqual(output.shape, [2, 8, 8, 3])
    }

    func testRandomVerticalFlipNever() {
        let flip = transforms.RandomVerticalFlip(p: 0.0)
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = flip(input)

        XCTAssertEqual(output.shape, [2, 8, 8, 3])
    }
}

// MARK: - Grayscale Tests

final class GrayscaleTransformTests: XCTestCase {

    func testGrayscaleRGBToGray() {
        let gray = transforms.Grayscale(numOutputChannels: 1)
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = gray(input)

        XCTAssertEqual(output.shape, [1, 8, 8, 1])
    }

    func testGrayscaleRGBToGray3Channel() {
        let gray = transforms.Grayscale(numOutputChannels: 3)
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = gray(input)

        XCTAssertEqual(output.shape, [1, 8, 8, 3])
    }

    func testGrayscaleAlreadyGray() {
        let gray = transforms.Grayscale(numOutputChannels: 1)
        let input = Tensor<Float>.ones([1, 8, 8, 1])
        let output = gray(input)

        XCTAssertEqual(output.shape, [1, 8, 8, 1])
    }

    func testRandomGrayscale() {
        let gray = transforms.RandomGrayscale(p: 0.5)
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = gray(input)

        XCTAssertEqual(output.shape, [1, 8, 8, 3])
    }
}

// MARK: - Invert Tests

final class InvertTransformTests: XCTestCase {

    func testRandomInvertAlways() {
        let invert = transforms.RandomInvert(p: 1.0)
        let input = Tensor<Float>.zeros([1, 4, 4, 3])
        let output = invert(input)

        XCTAssertEqual(output.shape, [1, 4, 4, 3])

        // 1 - 0 = 1
        let values = output.scalars()
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-5)
    }

    func testRandomInvertNever() {
        let invert = transforms.RandomInvert(p: 0.0)
        let input = Tensor<Float>.zeros([1, 4, 4, 3])
        let output = invert(input)

        let values = output.scalars()
        XCTAssertEqual(values[0], 0.0, accuracy: 1e-5)
    }
}

// MARK: - Lambda Tests

final class LambdaTransformTests: XCTestCase {

    func testLambdaIdentity() {
        let lambda = transforms.Lambda { $0 }
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = lambda(input)

        XCTAssertEqual(output.shape, input.shape)
    }

    func testLambdaMultiply() {
        let lambda = transforms.Lambda { $0 + $0 }  // Double the values
        let input = Tensor<Float>.ones([2, 4, 4, 1])
        let output = lambda(input)

        XCTAssertEqual(output.shape, input.shape)

        let values = output.scalars()
        XCTAssertEqual(values[0], 2.0, accuracy: 1e-5)
    }
}

// MARK: - Identity Tests

final class IdentityTransformTests: XCTestCase {

    func testIdentity() {
        let identity = transforms.Identity()
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = identity(input)

        XCTAssertEqual(output.shape, input.shape)
    }
}

// MARK: - Functional Tests

final class FunctionalTransformTests: XCTestCase {

    func testFunctionalNormalize() {
        let input = Tensor<Float>.ones([2, 8, 8, 1])
        let output = functional.normalize(input, mean: [0.5], std: [0.5])

        XCTAssertEqual(output.shape, input.shape)
    }

    func testFunctionalPad() {
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = functional.pad(input, padding: (1, 1, 1, 1))

        XCTAssertEqual(output.shape, [1, 10, 10, 3])
    }

    func testFunctionalCrop() {
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = functional.crop(input, top: 2, left: 2, height: 4, width: 4)

        XCTAssertEqual(output.shape, [1, 4, 4, 3])
    }

    func testFunctionalHFlip() {
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = functional.hflip(input)

        XCTAssertEqual(output.shape, input.shape)
    }

    func testFunctionalVFlip() {
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = functional.vflip(input)

        XCTAssertEqual(output.shape, input.shape)
    }
}

// MARK: - Integration Tests

final class TransformIntegrationTests: XCTestCase {

    func testImageNetPreprocessing() {
        // Standard ImageNet preprocessing pipeline
        let preprocess = transforms.Compose([
            transforms.CenterCrop(size: (224, 224)),
            transforms.Normalize(
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225]
            )
        ])

        let input = Tensor<Float>.randn([1, 256, 256, 3])
        let output = preprocess(input)

        XCTAssertEqual(output.shape, [1, 224, 224, 3])
    }

    func testMNISTPreprocessing() {
        // MNIST preprocessing
        let preprocess = transforms.Compose([
            transforms.Normalize(mean: [0.1307], std: [0.3081])
        ])

        let input = Tensor<Float>.randn([64, 28, 28, 1])
        let output = preprocess(input)

        XCTAssertEqual(output.shape, [64, 28, 28, 1])
    }

    func testDataAugmentationPipeline() {
        // Training augmentation pipeline
        let augment = transforms.Compose([
            transforms.RandomCrop(size: (28, 28), padding: (4, 4, 4, 4)),
            transforms.RandomHorizontalFlip(p: 0.5),
            transforms.Normalize(mean: [0.5], std: [0.5])
        ])

        let input = Tensor<Float>.randn([16, 28, 28, 1])
        let output = augment(input)

        XCTAssertEqual(output.shape, [16, 28, 28, 1])
    }
}
