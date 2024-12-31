import CoreML
import Foundation
@preconcurrency import Hub
import Tokenizers

// MARK: - TextGenerationPipeline

public class TextGenerationPipeline {
    // MARK: Lifecycle

    public convenience init(
        model: Model,
        prewarm: Bool = true,
        hubAPI: HubApi = .shared
    ) async throws {
        try await self.init(
            modelName: model.rawValue,
            prewarm: prewarm,
            hubAPI: hubAPI
        )
    }

    public init(
        modelName: String,
        prewarm: Bool = true,
        hubAPI: HubApi = .shared
    ) async throws {
        let signposter = Signposter(category: String(describing: TextGenerationPipeline.self))
        self.signposter = signposter

        async let _tokenizer = signposter.measure("Tokenizer Load") {
            try await AutoTokenizer.from(pretrained: modelName, hubApi: hubAPI)
        }

        async let _model = signposter.measure("Model Load") {
            try await CausalLMModel.from(pretrained: modelName, hubAPI: hubAPI)
        }

        (tokenizer, model) = try await (_tokenizer, _model)
        kvCache = model.makeKVCache()

        sampler = GreedySampler()

        if prewarm { try await self.prewarm() }
    }

    // MARK: Public

    public func prewarm() async throws {
        _ = try await signposter.measure("Prewarm") {
            try await model(
                inputIDs: MLShapedArray<Int32>(repeating: 0, shape: [1, 2]),
                causalMask: MLShapedArray<Float16>(repeating: 0, shape: [1, 1, 1, 2]),
                kvCache: model.makeKVCache()
            )
        }
    }

    // TODO: - assert/check prompt starts with last prompt
    public func generate(
        prompt: String,
        maxNewTokens: Int? = nil
    ) -> any AsyncSequence<String, Error> {
        generate(
            prompt: .text(prompt),
            maxNewTokens: maxNewTokens
        )
    }

    // TODO: - assert/check messages starts with last messages
    public func generate(
        messages: [[String: String]],
        maxNewTokens: Int? = nil
    ) -> any AsyncSequence<String, Error> {
        generate(
            prompt: .messages(messages),
            maxNewTokens: maxNewTokens
        )
    }

    public func callAsFunction(
        prompt: String,
        maxNewTokens: Int? = nil
    ) -> any AsyncSequence<String, Error> {
        generate(
            prompt: prompt,
            maxNewTokens: maxNewTokens
        )
    }

    public func callAsFunction(
        messages: [[String: String]],
        maxNewTokens: Int? = nil
    ) -> any AsyncSequence<String, Error> {
        generate(
            messages: messages,
            maxNewTokens: maxNewTokens
        )
    }

    // MARK: Private

    private let tokenizer: any Tokenizer
    private let model: CausalLMModel
    private let sampler: Sampler
    private let kvCache: CausalLMModel.KVCache

    private let signposter: Signposter
}

private extension TextGenerationPipeline {
    func generate(
        prompt: Prompt,
        maxNewTokens: Int? = nil
    ) -> any AsyncSequence<String, Error> {
        struct ModelContainer: @unchecked Sendable {
            let model: CausalLMModel
            let kvCache: CausalLMModel.KVCache
            let tokenizer: Tokenizer
            let sampler: Sampler
            let signposter: Signposter
        }
        let modelContainer = ModelContainer(
            model: model,
            kvCache: kvCache,
            tokenizer: tokenizer,
            sampler: sampler,
            signposter: signposter
        )
        return AsyncThrowingStream<String, Error> { continuation in
            Task {
                do {
                    let (model, kvCache, tokenizer, sampler, signposter) = (
                        modelContainer.model,
                        modelContainer.kvCache,
                        modelContainer.tokenizer,
                        modelContainer.sampler,
                        modelContainer.signposter
                    )
                    var tokens: [Int] = []
                    var tokensGenerated = 0
                    while true {
                        guard tokensGenerated < maxNewTokens ?? .max,
                              tokens.last == nil || tokens.last != tokenizer.eosTokenId else {
                            continuation.finish()
                            return
                        }

                        try Task.checkCancellation()

                        if tokens.isEmpty {
                            let promptInterval = signposter.begin("Prompt")

                            tokens = try signposter.measure("Tokenizer Encode") {
                                switch prompt {
                                case let .text(text):
                                    tokenizer.encode(text: text)
                                case let .messages(messages):
                                    try tokenizer.applyChatTemplate(
                                        messages: messages
                                    )
                                }
                            }

                            let inputIDs = MLShapedArray(scalars: tokens.map(Int32.init), shape: [1, tokens.count])

                            let causalMask = await signposter.measure("Create Causal Mask") {
                                await MLTensor.causalMask(size: tokens.count)
                                    .shapedArray(of: Float16.self)
                            }

                            let logits = try await signposter.measure("Prompt Prediction") {
                                try await model(
                                    inputIDs: inputIDs,
                                    causalMask: causalMask,
                                    kvCache: kvCache
                                )
                            }

                            let predictedToken = await signposter.measure("Sampling") {
                                await sampler.sample(logits[0, logits.shape[1] - 1])
                            }

                            tokens.append(predictedToken)
                            tokensGenerated += 1

                            let decodedText = signposter.measure("Tokenizer Decode") {
                                tokenizer.decode(tokens: [predictedToken], skipSpecialTokens: true)
                            }
                            continuation.yield(decodedText)

                            promptInterval.end(metadata: tokens.count)
                        } else {
                            try await signposter.measure("Extend") {
                                let inputIDs = MLShapedArray(scalars: [Int32(tokens.last!)], shape: [1, 1])
                                let causalMask = MLShapedArray<Float16>(
                                    repeating: 0,
                                    shape: [1, 1, 1, tokens.count]
                                )

                                let logits = try await signposter.measure("Extend Prediction") {
                                    try await model(
                                        inputIDs: inputIDs,
                                        causalMask: causalMask,
                                        kvCache: kvCache
                                    )
                                }

                                let predictedToken = await signposter.measure("Sampling") {
                                    await sampler.sample(logits[0, logits.shape[1] - 1])
                                }

                                tokens.append(predictedToken)
                                tokensGenerated += 1

                                let decodedText = signposter.measure("Tokenizer Decode") {
                                    tokenizer.decode(tokens: [predictedToken], skipSpecialTokens: true)
                                }
                                continuation.yield(decodedText)
                            }
                        }
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

// MARK: TextGenerationPipeline.Prompt

extension TextGenerationPipeline {
    enum Prompt {
        case text(String)
        case messages([[String: String]])
    }
}

// MARK: TextGenerationPipeline.Model

public extension TextGenerationPipeline {
    enum Model: String, CaseIterable {
        // MARK: - Llama

        case llama_3_2_1B_Instruct_4bit = "finnvoorhees/coreml-Llama-3.2-1B-Instruct-4bit"
        case llama_3_2_3B_Instruct_4bit = "finnvoorhees/coreml-Llama-3.2-3B-Instruct-4bit"

        // MARK: - Qwen

        case qwen2_5_0_5B_Instruct_4bit = "finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit"
        case qwen2_5_3B_Instruct_4bit = "finnvoorhees/coreml-Qwen2.5-3B-Instruct-4bit"
        case qwen2_5_Coder_0_5B_Instruct_4bit = "finnvoorhees/coreml-Qwen2.5-Coder-0.5B-Instruct-4bit"
        case qwen2_5_Coder_1_5B_Instruct_4bit = "finnvoorhees/coreml-Qwen2.5-Coder-1.5B-Instruct-4bit"

        // MARK: - SmolLM2

        case smolLM2_135M_Instruct_4bit = "finnvoorhees/coreml-SmolLM2-135M-Instruct-4bit"
        case smolLM2_135M_Instruct_8bit = "finnvoorhees/coreml-SmolLM2-135M-Instruct-8bit"
        case smolLM2_1_7B_Instruct_4bit = "finnvoorhees/coreml-SmolLM2-1.7B-Instruct-4bit"
        case smolLM2_360M_Instruct_4bit = "finnvoorhees/coreml-SmolLM2-360M-Instruct-4bit"
        case smolLM2_360M_Instruct_8bit = "finnvoorhees/coreml-SmolLM2-360M-Instruct-8bit"
        case smolLM2_1_7B_Instruct_8bit = "finnvoorhees/coreml-SmolLM2-1.7B-Instruct-8bit"
    }
}
