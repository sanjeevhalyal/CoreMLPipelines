import CoreML

protocol Sampler {
    func sample(_ logits: any MLShapedArrayProtocol<Float16>) async -> Int
}
