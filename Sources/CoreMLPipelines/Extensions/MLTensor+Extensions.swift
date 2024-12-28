import CoreML

extension MLTensor {
    static func causalMask(size: Int) -> MLTensor {
        (
            MLTensor(
                repeating: Float16(1),
                shape: [1, 1, size, size]
            ).bandPart(
                lowerBandCount: -1,
                upperBandCount: 0
            ) - 1
        ) * Float16.greatestFiniteMagnitude
    }
}
