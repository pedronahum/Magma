// Magma - Transform Tests
// Tests for PyTorch-compatible data transforms

import Testing
@testable import Magma
@testable import LazyTensor

// MARK: - Compose Tests

@Suite("Compose Transform Tests")
struct ComposeTransformTests {

    @Test("Compose empty list")
    func composeEmptyList() {
        let compose = transforms.Compose([])
        let input = Tensor<Float>.ones([2, 4, 4, 3])
        let output = compose(input)
        #expect(output.shape == input.shape)
    }

    @Test("Compose single transform")
    func composeSingleTransform() {
        let compose = transforms.Compose([
            transforms.Identity()
        ])
        let input = Tensor<Float>.ones([2, 4, 4, 3])
        let output = compose(input)
        #expect(output.shape == input.shape)
    }

    @Test("Compose multiple transforms")
    func composeMultipleTransforms() {
        let compose = transforms.Compose([
            transforms.Normalize(mean: [0.5], std: [0.5]),
            transforms.Identity()
        ])
        let input = Tensor<Float>.ones([2, 28, 28, 1])
        let output = compose(input)
        #expect(output.shape == [2, 28, 28, 1])
    }
}

// MARK: - Normalize Tests

@Suite("Normalize Transform Tests")
struct NormalizeTransformTests {

    @Test("Normalize single channel")
    func normalizeSingleChannel() {
        let normalize = transforms.Normalize(mean: [0.5], std: [0.5])
        let input = Tensor<Float>.ones([2, 28, 28, 1])
        let output = normalize(input)

        #expect(output.shape == [2, 28, 28, 1])

        // (1.0 - 0.5) / 0.5 = 1.0
        let values = output.scalars()
        #expect(abs(values[0] - 1.0) < 1e-5)
    }

    @Test("Normalize multi channel")
    func normalizeMultiChannel() {
        let normalize = transforms.Normalize(
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225]
        )
        let input = Tensor<Float>.ones([1, 4, 4, 3])
        let output = normalize(input)

        #expect(output.shape == [1, 4, 4, 3])
    }

    @Test("Normalize 2D")
    func normalize2D() {
        // For [batch, features] input
        let normalize = transforms.Normalize(mean: [0.0], std: [1.0])
        let input = Tensor<Float>.randn([32, 10])
        let output = normalize(input)

        #expect(output.shape == [32, 10])
    }

    @Test("Denormalize")
    func denormalize() {
        let mean: [Float] = [0.5, 0.5, 0.5]
        let std: [Float] = [0.25, 0.25, 0.25]

        let normalize = transforms.Normalize(mean: mean, std: std)
        let denormalize = transforms.Denormalize(mean: mean, std: std)

        let input = Tensor<Float>.randn([1, 8, 8, 3])
        let normalized = normalize(input)
        let recovered = denormalize(normalized)

        #expect(recovered.shape == input.shape)
    }
}

// MARK: - RandomApply Tests

@Suite("RandomApply Transform Tests")
struct RandomApplyTransformTests {

    @Test("RandomApply always")
    func randomApplyAlways() {
        // p=1.0 should always apply
        let apply = transforms.RandomApply([
            transforms.Normalize(mean: [0.5], std: [0.5])
        ], p: 1.0)

        let input = Tensor<Float>.ones([2, 8, 8, 1])
        let output = apply(input)

        #expect(output.shape == input.shape)
    }

    @Test("RandomApply never")
    func randomApplyNever() {
        // p=0.0 should never apply
        let apply = transforms.RandomApply([
            transforms.Normalize(mean: [0.5], std: [0.5])
        ], p: 0.0)

        let input = Tensor<Float>.ones([2, 8, 8, 1])
        let output = apply(input)

        #expect(output.shape == input.shape)
        // Should be unchanged
        let values = output.scalars()
        #expect(abs(values[0] - 1.0) < 1e-5)
    }
}

// MARK: - RandomChoice Tests

@Suite("RandomChoice Transform Tests")
struct RandomChoiceTransformTests {

    @Test("RandomChoice single option")
    func randomChoiceSingleOption() {
        let choice = transforms.RandomChoice([
            transforms.Identity()
        ])

        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = choice(input)

        #expect(output.shape == input.shape)
    }

    @Test("RandomChoice multiple options")
    func randomChoiceMultipleOptions() {
        let choice = transforms.RandomChoice([
            transforms.Identity(),
            transforms.Normalize(mean: [0.5], std: [0.5])
        ])

        let input = Tensor<Float>.ones([2, 8, 8, 1])
        let output = choice(input)

        #expect(output.shape == input.shape)
    }
}

// MARK: - CenterCrop Tests

@Suite("CenterCrop Transform Tests")
struct CenterCropTransformTests {

    @Test("CenterCrop 4D")
    func centerCrop4D() {
        let crop = transforms.CenterCrop(size: (4, 4))
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = crop(input)

        #expect(output.shape == [2, 4, 4, 3])
    }

    @Test("CenterCrop 3D")
    func centerCrop3D() {
        let crop = transforms.CenterCrop(size: (6, 6))
        let input = Tensor<Float>.ones([10, 10, 3])
        let output = crop(input)

        #expect(output.shape == [6, 6, 3])
    }

    @Test("CenterCrop square")
    func centerCropSquare() {
        let crop = transforms.CenterCrop(size: 14)
        let input = Tensor<Float>.ones([1, 28, 28, 1])
        let output = crop(input)

        #expect(output.shape == [1, 14, 14, 1])
    }

    @Test("CenterCrop same size")
    func centerCropSameSize() {
        let crop = transforms.CenterCrop(size: (8, 8))
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = crop(input)

        #expect(output.shape == [1, 8, 8, 3])
    }
}

// MARK: - RandomCrop Tests

@Suite("RandomCrop Transform Tests")
struct RandomCropTransformTests {

    @Test("RandomCrop 4D")
    func randomCrop4D() {
        let crop = transforms.RandomCrop(size: (4, 4))
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = crop(input)

        #expect(output.shape == [2, 4, 4, 3])
    }

    @Test("RandomCrop 3D")
    func randomCrop3D() {
        let crop = transforms.RandomCrop(size: (6, 6))
        let input = Tensor<Float>.ones([10, 10, 3])
        let output = crop(input)

        #expect(output.shape == [6, 6, 3])
    }

    @Test("RandomCrop square")
    func randomCropSquare() {
        let crop = transforms.RandomCrop(size: 14)
        let input = Tensor<Float>.ones([1, 28, 28, 1])
        let output = crop(input)

        #expect(output.shape == [1, 14, 14, 1])
    }
}

// MARK: - Pad Tests

@Suite("Pad Transform Tests")
struct PadTransformTests {

    @Test("Pad uniform")
    func padUniform() {
        let pad = transforms.Pad(padding: 2)
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = pad(input)

        #expect(output.shape == [1, 12, 12, 3])
    }

    @Test("Pad non-uniform")
    func padNonUniform() {
        let pad = transforms.Pad(padding: (1, 2, 3, 4))  // top, bottom, left, right
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = pad(input)

        #expect(output.shape == [1, 11, 15, 3])  // 8+1+2=11, 8+3+4=15
    }

    @Test("Pad 3D")
    func pad3D() {
        let pad = transforms.Pad(padding: 1)
        let input = Tensor<Float>.ones([8, 8, 3])
        let output = pad(input)

        #expect(output.shape == [10, 10, 3])
    }
}

// MARK: - Flip Tests

@Suite("Flip Transform Tests")
struct FlipTransformTests {

    @Test("RandomHorizontalFlip always")
    func randomHorizontalFlipAlways() {
        // Not easily testable due to randomness, but check shape
        let flip = transforms.RandomHorizontalFlip(p: 1.0)
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = flip(input)

        #expect(output.shape == [2, 8, 8, 3])
    }

    @Test("RandomHorizontalFlip never")
    func randomHorizontalFlipNever() {
        let flip = transforms.RandomHorizontalFlip(p: 0.0)
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = flip(input)

        #expect(output.shape == [2, 8, 8, 3])
    }

    @Test("RandomVerticalFlip always")
    func randomVerticalFlipAlways() {
        let flip = transforms.RandomVerticalFlip(p: 1.0)
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = flip(input)

        #expect(output.shape == [2, 8, 8, 3])
    }

    @Test("RandomVerticalFlip never")
    func randomVerticalFlipNever() {
        let flip = transforms.RandomVerticalFlip(p: 0.0)
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = flip(input)

        #expect(output.shape == [2, 8, 8, 3])
    }
}

// MARK: - Grayscale Tests

@Suite("Grayscale Transform Tests")
struct GrayscaleTransformTests {

    @Test("Grayscale RGB to gray")
    func grayscaleRGBToGray() {
        let gray = transforms.Grayscale(numOutputChannels: 1)
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = gray(input)

        #expect(output.shape == [1, 8, 8, 1])
    }

    @Test("Grayscale RGB to gray 3 channel")
    func grayscaleRGBToGray3Channel() {
        let gray = transforms.Grayscale(numOutputChannels: 3)
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = gray(input)

        #expect(output.shape == [1, 8, 8, 3])
    }

    @Test("Grayscale already gray")
    func grayscaleAlreadyGray() {
        let gray = transforms.Grayscale(numOutputChannels: 1)
        let input = Tensor<Float>.ones([1, 8, 8, 1])
        let output = gray(input)

        #expect(output.shape == [1, 8, 8, 1])
    }

    @Test("RandomGrayscale")
    func randomGrayscale() {
        let gray = transforms.RandomGrayscale(p: 0.5)
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = gray(input)

        #expect(output.shape == [1, 8, 8, 3])
    }
}

// MARK: - Invert Tests

@Suite("Invert Transform Tests")
struct InvertTransformTests {

    @Test("RandomInvert always")
    func randomInvertAlways() {
        let invert = transforms.RandomInvert(p: 1.0)
        let input = Tensor<Float>.zeros([1, 4, 4, 3])
        let output = invert(input)

        #expect(output.shape == [1, 4, 4, 3])

        // 1 - 0 = 1
        let values = output.scalars()
        #expect(abs(values[0] - 1.0) < 1e-5)
    }

    @Test("RandomInvert never")
    func randomInvertNever() {
        let invert = transforms.RandomInvert(p: 0.0)
        let input = Tensor<Float>.zeros([1, 4, 4, 3])
        let output = invert(input)

        let values = output.scalars()
        #expect(abs(values[0] - 0.0) < 1e-5)
    }
}

// MARK: - Lambda Tests

@Suite("Lambda Transform Tests")
struct LambdaTransformTests {

    @Test("Lambda identity")
    func lambdaIdentity() {
        let lambda = transforms.Lambda { $0 }
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = lambda(input)

        #expect(output.shape == input.shape)
    }

    @Test("Lambda multiply")
    func lambdaMultiply() {
        let lambda = transforms.Lambda { $0 + $0 }  // Double the values
        let input = Tensor<Float>.ones([2, 4, 4, 1])
        let output = lambda(input)

        #expect(output.shape == input.shape)

        let values = output.scalars()
        #expect(abs(values[0] - 2.0) < 1e-5)
    }
}

// MARK: - Identity Tests

@Suite("Identity Transform Tests")
struct IdentityTransformTests {

    @Test("Identity")
    func identity() {
        let identity = transforms.Identity()
        let input = Tensor<Float>.ones([2, 8, 8, 3])
        let output = identity(input)

        #expect(output.shape == input.shape)
    }
}

// MARK: - Functional Tests

@Suite("Functional Transform Tests")
struct FunctionalTransformTests {

    @Test("Functional normalize")
    func functionalNormalize() {
        let input = Tensor<Float>.ones([2, 8, 8, 1])
        let output = functional.normalize(input, mean: [0.5], std: [0.5])

        #expect(output.shape == input.shape)
    }

    @Test("Functional pad")
    func functionalPad() {
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = functional.pad(input, padding: (1, 1, 1, 1))

        #expect(output.shape == [1, 10, 10, 3])
    }

    @Test("Functional crop")
    func functionalCrop() {
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = functional.crop(input, top: 2, left: 2, height: 4, width: 4)

        #expect(output.shape == [1, 4, 4, 3])
    }

    @Test("Functional hflip")
    func functionalHFlip() {
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = functional.hflip(input)

        #expect(output.shape == input.shape)
    }

    @Test("Functional vflip")
    func functionalVFlip() {
        let input = Tensor<Float>.ones([1, 8, 8, 3])
        let output = functional.vflip(input)

        #expect(output.shape == input.shape)
    }
}

// MARK: - Integration Tests

@Suite("Transform Integration Tests")
struct TransformIntegrationTests {

    @Test("ImageNet preprocessing")
    func imageNetPreprocessing() {
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

        #expect(output.shape == [1, 224, 224, 3])
    }

    @Test("MNIST preprocessing")
    func mnistPreprocessing() {
        // MNIST preprocessing
        let preprocess = transforms.Compose([
            transforms.Normalize(mean: [0.1307], std: [0.3081])
        ])

        let input = Tensor<Float>.randn([64, 28, 28, 1])
        let output = preprocess(input)

        #expect(output.shape == [64, 28, 28, 1])
    }

    @Test("Data augmentation pipeline")
    func dataAugmentationPipeline() {
        // Training augmentation pipeline
        let augment = transforms.Compose([
            transforms.RandomCrop(size: (28, 28), padding: (4, 4, 4, 4)),
            transforms.RandomHorizontalFlip(p: 0.5),
            transforms.Normalize(mean: [0.5], std: [0.5])
        ])

        let input = Tensor<Float>.randn([16, 28, 28, 1])
        let output = augment(input)

        #expect(output.shape == [16, 28, 28, 1])
    }
}
