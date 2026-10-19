import CoreML
import Foundation

/// Use from an iPhone app with preprocessed patch feature providers.
/// Model loading, image preprocessing and cross-patch box merging are excluded.
@available(iOS 16.0, macOS 13.0, *)
public final class CoreMLPatchBenchmark {
    public struct Result: Codable {
        public let computeUnits: String
        public let iterations: Int
        public let totalSeconds: Double
        public let medianMilliseconds: Double
        public let p95Milliseconds: Double
        public let samplesMilliseconds: [Double]
    }

    public enum BenchmarkError: Error {
        case invalidArguments
    }

    private let model: MLModel
    private let computeUnits: MLComputeUnits
    private let decoder: MLModel?
    private let lock = NSLock()

    public init(compiledModelURL: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        self.computeUnits = computeUnits
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        model = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
        decoder = nil
    }

    /// The feature model runs with CPU/ANE and the decoder with CPU/GPU.
    /// Forward MLFeatureValue objects directly, without copying feature arrays.
    public init(featuresModelURL: URL, decoderModelURL: URL) throws {
        computeUnits = .cpuAndNeuralEngine
        let featureConfiguration = MLModelConfiguration()
        featureConfiguration.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: featuresModelURL, configuration: featureConfiguration)
        let decoderConfiguration = MLModelConfiguration()
        decoderConfiguration.computeUnits = .cpuAndGPU
        decoder = try MLModel(contentsOf: decoderModelURL, configuration: decoderConfiguration)
    }

    private func forward(_ input: MLFeatureProvider) throws -> MLFeatureProvider {
        let features = try model.prediction(from: input)
        if let decoder {
            return try decoder.prediction(from: features)
        }
        return features
    }

    /// Raw pred_logits and normalized cxcywh pred_boxes for detection checks.
    public func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider {
        lock.lock()
        defer { lock.unlock() }
        return try forward(input)
    }

    /// Pass 50 different 640x640 patches to measure the intended workload.
    /// Keep this runner alive across images; do not reload the model per patch.
    public func run(inputs: [MLFeatureProvider], iterations: Int = 50, warmup: Int = 3) throws -> Result {
        guard !inputs.isEmpty, iterations > 0, warmup >= 0 else {
            throw BenchmarkError.invalidArguments
        }
        lock.lock()
        defer { lock.unlock() }
        for index in 0..<warmup {
            try autoreleasepool {
                _ = try forward(inputs[index % inputs.count])
            }
        }
        var milliseconds: [Double] = []
        for index in 0..<iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            try autoreleasepool {
                _ = try forward(inputs[index % inputs.count])
            }
            milliseconds.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
        }
        let sorted = milliseconds.sorted()
        let median = (sorted[(iterations - 1) / 2] + sorted[iterations / 2]) / 2
        let p95Index = Int(ceil(Double(iterations - 1) * 0.95))
        let units = decoder == nil ? String(describing: computeUnits) : "features: CPU/ANE; decoder: CPU/GPU"
        return Result(computeUnits: units, iterations: iterations,
                      totalSeconds: milliseconds.reduce(0, +) / 1000,
                      medianMilliseconds: median, p95Milliseconds: sorted[p95Index],
                      samplesMilliseconds: milliseconds)
    }
}
