// Magma - Global seed tests
//
// `manualSeed(_:)` makes every host-side random draw reproducible: tensor
// factories, initializers, layer init, Dropout masks, loader shuffles and
// random transforms. Draws from the global stream are process-wide, so the
// suite is serialized and each test reseeds before every draw it compares.

import Testing
@testable import Magma
@testable import LazyTensor

@Suite("Global random seed", .serialized)
struct RandomSeedTests {

    @Test("randn: same seed gives identical values, different seeds differ")
    func randnSeeded() {
        manualSeed(42)
        let a = Tensor<Float>.randn([5, 7]).scalars()
        manualSeed(42)
        let b = Tensor<Float>.randn([5, 7]).scalars()
        manualSeed(43)
        let c = Tensor<Float>.randn([5, 7]).scalars()
        #expect(a.count == 35)
        #expect(a == b)
        #expect(a != c)
    }

    @Test("A seeded stream advances: consecutive draws differ")
    func seededStreamAdvances() {
        manualSeed(7)
        let a = Tensor<Float>.randn([16]).scalars()
        let b = Tensor<Float>.randn([16]).scalars()
        #expect(a != b)
    }

    @Test("uniform is seeded and stays in the half-open range [low, high)")
    func uniformSeededHalfOpen() {
        manualSeed(1)
        let a = Tensor<Float>.uniform(low: -0.5, high: 0.25, shape: [1000]).scalars()
        manualSeed(1)
        let b = Tensor<Float>.uniform(low: -0.5, high: 0.25, shape: [1000]).scalars()
        #expect(a == b)
        #expect(a.allSatisfy { $0 >= -0.5 && $0 < 0.25 })
    }

    @Test("Initializers are seeded; truncatedNormal stays within 2 stddev")
    func initializersSeeded() {
        manualSeed(3)
        let g1 = Tensor<Float>.glorotUniform([8, 4]).scalars()
        let t1 = Tensor<Float>.truncatedNormal([500], mean: 1, stddev: 0.5).scalars()
        manualSeed(3)
        let g2 = Tensor<Float>.glorotUniform([8, 4]).scalars()
        let t2 = Tensor<Float>.truncatedNormal([500], mean: 1, stddev: 0.5).scalars()
        #expect(g1 == g2)
        #expect(t1 == t2)
        #expect(t1.count == 500)
        #expect(t1.allSatisfy { $0 >= 0 && $0 <= 2 })
    }

    @Test("nn.Linear gets identical weights under the same seed")
    func linearInitSeeded() {
        manualSeed(2024)
        let l1 = nn.Linear(inputSize: 6, outputSize: 3)
        manualSeed(2024)
        let l2 = nn.Linear(inputSize: 6, outputSize: 3)
        manualSeed(2025)
        let l3 = nn.Linear(inputSize: 6, outputSize: 3)
        let w1 = l1.parameters().flatMap { $0.value.scalars() }
        let w2 = l2.parameters().flatMap { $0.value.scalars() }
        let w3 = l3.parameters().flatMap { $0.value.scalars() }
        #expect(!w1.isEmpty)
        #expect(w1 == w2)
        #expect(w1 != w3)
    }

    @Test("Dropout masks are reproducible under a seed")
    func dropoutSeeded() {
        let x = Tensor<Float>.ones([4, 64])
        let dropout = nn.Dropout(p: 0.5)
        manualSeed(11)
        let a = dropout(x).scalars()
        manualSeed(11)
        let b = dropout(x).scalars()
        manualSeed(12)
        let c = dropout(x).scalars()
        #expect(a.count == 256)
        #expect(a == b)
        #expect(a != c)
        // Inverted dropout: every element is either dropped or scaled by 2.
        #expect(a.allSatisfy { $0 == 0 || abs($0 - 2) < 1e-6 })
    }

    @Test("RandomCrop picks the same location under the same seed")
    func randomCropSeeded() {
        let image = Tensor<Float>((0..<64).map(Float.init), shape: [8, 8, 1])
        let crop = transforms.RandomCrop(size: 3)
        manualSeed(5)
        let a = crop(image).scalars()
        manualSeed(5)
        let b = crop(image).scalars()
        #expect(a.count == 9)
        #expect(a == b)
    }
}
