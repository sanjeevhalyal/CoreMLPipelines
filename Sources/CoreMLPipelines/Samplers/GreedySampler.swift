import CoreML

struct GreedySampler: Sampler {
    func sample(_ logits: any MLShapedArrayProtocol<Float16>) async -> Int {
        await Int(MLTensor(logits).argmax().shapedArray(of: Int32.self).scalar!)
    }
}
