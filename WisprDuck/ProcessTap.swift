import Foundation
import CoreAudio
import os

private let logger = Logger(subsystem: "com.wisprduck", category: "ProcessTap")

/// Manages a single Core Audio process tap: intercepts one process's audio output,
/// scales it by a duck factor, and plays it to the real output device.
/// Requires Screen & System Audio Recording permission and NSAudioCaptureUsageDescription.
///
/// Lifecycle: init → start() → updateDuckLevel() → stop() → deinit
/// Cleanup order: AudioDeviceStop → DestroyIOProcID → DestroyAggregateDevice → DestroyProcessTap
final class ProcessTap {
    let processObjectID: AudioObjectID
    let pid: pid_t
    private(set) var lastError: String?

    private var tapID: AudioObjectID = kAudioObjectUnknown
    private var aggregateDeviceID: AudioObjectID = kAudioObjectUnknown
    private var ioProcID: AudioDeviceIOProcID?
    private let tapUUID = UUID()
    private let aggregateUUID = UUID()
    private let ioQueue = DispatchQueue(label: "com.wisprduck.processtap.io", qos: .userInitiated)
    private var isRunning = false
    private var tapFormatIsFloat32 = true

    private var targetLevel: Float = 1.0
    private var currentLevel: Float = 1.0
    private var rampRate: Float = 0.0 // Max change per sample (linear ramp)
    // One filter history per output channel, allocated before audio I/O starts.
    private var lowPassA: [Float] = []
    private var lowPassB: [Float] = []
    private var lowPassCoefficient: Float = 1
    private var filterMix: Float = 0
    private var targetFilterMix: Float = 1
    private var filterMixRate: Float = 1
    private var sampleRate: Float = 44100
    private var targetLowPassCoefficient: Float = 1
    private var coefficientStep: Float = 0

    init(processObjectID: AudioObjectID, pid: pid_t) {
        self.processObjectID = processObjectID
        self.pid = pid
    }

    deinit {
        stop()
    }

    // MARK: - Public API

    /// Start intercepting audio. Returns true on success.
    /// - Parameters:
    ///   - outputDeviceUID: UID of the output device to route audio through
    ///   - duckLevel: Volume factor 0.0–1.0 (e.g., 0.2 for 20%)
    func start(outputDeviceUID: String, duckLevel: Float, cutoff: Float, blurMix: Float) -> Bool {
        guard !isRunning else { return true }
        lastError = nil

        let clampedLevel = max(0.0, min(1.0, duckLevel))
        targetLevel = clampedLevel
        currentLevel = clampedLevel // Start at duck level — no ramp on duck-in to avoid silence→pop
        targetFilterMix = max(0, min(1, blurMix))

        // 1. Create tap description
        let tapDesc = CATapDescription(stereoMixdownOfProcesses: [processObjectID])
        tapDesc.uuid = tapUUID
        tapDesc.name = "WisprDuck-\(pid)"
        tapDesc.muteBehavior = .mutedWhenTapped
        tapDesc.isPrivate = true

        // 2. Create process tap
        var status = AudioHardwareCreateProcessTap(tapDesc, &tapID)
        guard status == noErr else {
            let message = "Could not create system audio tap for PID \(pid): \(describeOSStatus(status))"
            lastError = message
            logger.error("\(message)")
            return false
        }

        // 3. Compute linear ramp rate from tap's sample rate
        rampRate = computeRampRate(tapID: tapID)
        lowPassCoefficient = 1 - exp(-2 * .pi * max(300, min(6000, cutoff)) / sampleRate)
        targetLowPassCoefficient = lowPassCoefficient

        // 4. Create aggregate device combining real output + tap
        let aggDesc: [String: Any] = [
            kAudioAggregateDeviceNameKey: "WisprDuck-Agg-\(pid)",
            kAudioAggregateDeviceUIDKey: aggregateUUID.uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputDeviceUID,
            kAudioAggregateDeviceClockDeviceKey: outputDeviceUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [
                [
                    kAudioSubDeviceUIDKey: outputDeviceUID,
                    kAudioSubDeviceDriftCompensationKey: false,
                ]
            ],
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapUUID.uuidString,
                    kAudioSubTapDriftCompensationKey: true,
                ]
            ],
        ]

        status = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggregateDeviceID)
        guard status == noErr else {
            let message = "Could not create aggregate audio device for PID \(pid): \(describeOSStatus(status))"
            lastError = message
            logger.error("\(message)")
            cleanupTap()
            return false
        }

        guard prepareOutputChannels() else {
            lastError = "Could not read aggregate device output channels for PID \(pid)"
            cleanupAggregateDevice()
            cleanupTap()
            return false
        }

        // 5. Create IO proc (block-based, dispatched to our serial queue)
        status = AudioDeviceCreateIOProcIDWithBlock(&ioProcID, aggregateDeviceID, ioQueue) {
            [self] _, inInputData, _, outOutputData, _ in
            // This block captures self strongly. ProcessTap.stop() must be called
            // before deallocation to break the cycle (stop destroys the IO proc).
            self.processAudioBuffers(input: inInputData, output: outOutputData)
        }
        guard status == noErr else {
            let message = "Could not create audio IO callback for PID \(pid): \(describeOSStatus(status))"
            lastError = message
            logger.error("\(message)")
            cleanupAggregateDevice()
            cleanupTap()
            return false
        }

        // 6. Start the device
        status = AudioDeviceStart(aggregateDeviceID, ioProcID)
        guard status == noErr else {
            let message = "Could not start aggregate audio device for PID \(pid): \(describeOSStatus(status))"
            lastError = message
            logger.error("\(message)")
            cleanupIOProc()
            cleanupAggregateDevice()
            cleanupTap()
            return false
        }

        isRunning = true
        return true
    }

    /// Stop intercepting audio. Cleans up all Core Audio resources.
    /// Safe to call multiple times.
    func stop() {
        guard isRunning else { return }
        isRunning = false

        // Strict cleanup order: Stop → DestroyIOProc → DestroyAggregate → DestroyTap
        if let procID = ioProcID {
            let status = AudioDeviceStop(aggregateDeviceID, procID)
            if status != noErr {
                logger.error("Could not stop aggregate audio device for PID \(self.pid): \(describeOSStatus(status))")
            }
        }
        cleanupIOProc()
        cleanupAggregateDevice()
        cleanupTap()
    }

    /// Update gain independently of EQ (100% gain can still be blurred).
    func updateDuckLevel(_ level: Float) {
        let clampedLevel = max(0.0, min(1.0, level))
        ioQueue.async { [weak self] in self?.targetLevel = clampedLevel }
    }

    func updateFilter(cutoff: Float, mix: Float) {
        ioQueue.async { [weak self] in
            guard let self else { return }
            let frequency = max(300, min(6000, cutoff))
            self.targetLowPassCoefficient = 1 - exp(-2 * .pi * frequency / self.sampleRate)
            self.coefficientStep = (self.targetLowPassCoefficient - self.lowPassCoefficient) / (self.sampleRate * 0.15)
            self.targetFilterMix = max(0, min(1, mix))
        }
    }

    func restoreNormal() {
        ioQueue.async { [weak self] in
            self?.targetLevel = 1
            self?.targetFilterMix = 0
        }
    }

    // MARK: - Audio Processing

    /// Scales and low-pass filters Float32 stereo samples. Each channel keeps
    /// independent filter history across callbacks; the dry/wet mix ramps to
    /// avoid a treble jump when the tap starts or stops.
    ///
    /// The aggregate device's input buffer layout is:
    ///   [output device's input buffers...] [tap's input buffers...]
    /// If the output device has input channels (e.g. Scarlett 2i2 has mic inputs),
    /// the tap's audio starts AFTER those buffers. We must offset into the correct
    /// position to read the tapped audio rather than the device's mic input.
    private func processAudioBuffers(
        input inInputData: UnsafePointer<AudioBufferList>,
        output outOutputData: UnsafeMutablePointer<AudioBufferList>
    ) {
        let inputs = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
        let outputs = UnsafeMutableAudioBufferListPointer(outOutputData)

        // Tap buffers are at the END of the input list, after the output device's own inputs.
        let tapOffset = max(0, inputs.count - outputs.count)

        guard tapFormatIsFloat32 else {
            // Pass-through without scaling if the format is not Float32 PCM.
            for (i, output) in outputs.enumerated() {
                let inputIndex = tapOffset + i
                guard inputIndex < inputs.count,
                      let inData = inputs[inputIndex].mData,
                      let outData = output.mData else {
                    if let outData = output.mData {
                        memset(outData, 0, Int(output.mDataByteSize))
                    }
                    continue
                }
                let bytes = min(Int(inputs[inputIndex].mDataByteSize), Int(output.mDataByteSize))
                memcpy(outData, inData, bytes)
            }
            return
        }

        let target = targetLevel
        let startingLevel = currentLevel
        var endingLevel = startingLevel
        let filterStep = filterMixRate
        let targetMix = targetFilterMix
        let startingMix = filterMix
        var endingMix = startingMix
        let rate = rampRate
        let targetAlpha = targetLowPassCoefficient
        let alphaStep = coefficientStep
        var endingAlpha = lowPassCoefficient

        var channelBase = 0
        for (i, output) in outputs.enumerated() {
            let inputIndex = tapOffset + i
            guard inputIndex < inputs.count,
                  let inData = inputs[inputIndex].mData,
                  let outData = output.mData else {
                if let outData = output.mData {
                    memset(outData, 0, Int(output.mDataByteSize))
                }
                continue
            }

            let inSamples = inData.assumingMemoryBound(to: Float.self)
            let outSamples = outData.assumingMemoryBound(to: Float.self)
            let byteCount = min(Int(inputs[inputIndex].mDataByteSize), Int(output.mDataByteSize))
            let sampleCount = byteCount / MemoryLayout<Float>.size

            let channels = Int(output.mNumberChannels)
            guard channels > 0 else {
                memset(outData, 0, Int(output.mDataByteSize))
                continue
            }
            let base = channelBase
            channelBase += channels
            let frames = sampleCount / channels
            var current = startingLevel
            var filterAlpha = lowPassCoefficient
            var mix = startingMix
            for frame in 0..<frames {
                current += max(-rate, min(rate, target - current))
                filterAlpha += max(-abs(alphaStep), min(abs(alphaStep), targetAlpha - filterAlpha))
                mix += max(-filterStep, min(filterStep, targetMix - mix))
                for channel in 0..<channels {
                    let index = frame * channels + channel
                    let stateIndex = base + channel
                    let dry = inSamples[index]
                    if stateIndex < lowPassA.count {
                        lowPassA[stateIndex] += filterAlpha * (dry - lowPassA[stateIndex])
                        lowPassB[stateIndex] += filterAlpha * (lowPassA[stateIndex] - lowPassB[stateIndex])
                        outSamples[index] = (dry + mix * (lowPassB[stateIndex] - dry)) * current
                    } else {
                        // A device layout changed without a restart: still duck, never leak full-volume audio.
                        outSamples[index] = dry * current
                    }
                }
            }
            endingLevel = current
            endingAlpha = filterAlpha
            endingMix = mix
        }

        currentLevel = endingLevel
        filterMix = endingMix
        lowPassCoefficient = endingAlpha
    }

    // MARK: - Cleanup Helpers

    private func cleanupIOProc() {
        guard let procID = ioProcID else { return }
        let status = AudioDeviceDestroyIOProcID(aggregateDeviceID, procID)
        if status != noErr {
            logger.error("Could not destroy audio IO callback for PID \(self.pid): \(describeOSStatus(status))")
        }
        ioProcID = nil
    }

    private func cleanupAggregateDevice() {
        guard aggregateDeviceID != kAudioObjectUnknown else { return }
        let status = AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
        if status != noErr {
            logger.error("Could not destroy aggregate audio device for PID \(self.pid): \(describeOSStatus(status))")
        }
        aggregateDeviceID = kAudioObjectUnknown
    }

    private func cleanupTap() {
        guard tapID != kAudioObjectUnknown else { return }
        let status = AudioHardwareDestroyProcessTap(tapID)
        if status != noErr {
            logger.error("Could not destroy process tap for PID \(self.pid): \(describeOSStatus(status))")
        }
        tapID = kAudioObjectUnknown
    }

    // MARK: - Helpers

    /// Query the aggregate's output layout once, before starting real-time I/O.
    private func prepareOutputChannels() -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(aggregateDeviceID, &address, 0, nil, &size) == noErr,
              size >= MemoryLayout<AudioBufferList>.size else { return false }
        let storage = UnsafeMutableRawPointer.allocate(
            byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment
        )
        defer { storage.deallocate() }
        guard AudioObjectGetPropertyData(aggregateDeviceID, &address, 0, nil, &size, storage) == noErr else {
            return false
        }
        let buffers = UnsafeMutableAudioBufferListPointer(storage.assumingMemoryBound(to: AudioBufferList.self))
        let channels = buffers.reduce(0) { $0 + Int($1.mNumberChannels) }
        guard channels > 0 else { return false }
        lowPassA = [Float](repeating: 0, count: channels)
        lowPassB = [Float](repeating: 0, count: channels)
        return true
    }

    /// Compute the linear ramp rate (max volume change per sample) for 1-second transitions.
    /// At 48kHz: rate = 1/48000 ≈ 0.00002. A full 0→1 sweep takes exactly 1s.
    /// Partial sweeps are proportional (e.g., 0.1→1.0 takes 0.9s).
    private func computeRampRate(tapID: AudioObjectID) -> Float {
        var formatAddress = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var format = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)

        let status = AudioObjectGetPropertyData(tapID, &formatAddress, 0, nil, &size, &format)
        tapFormatIsFloat32 = status == noErr
            && format.mFormatID == kAudioFormatLinearPCM
            && (format.mFormatFlags & kAudioFormatFlagIsFloat) != 0
            && format.mBitsPerChannel == 32
        sampleRate = (status == noErr && format.mSampleRate > 0)
            ? Float(format.mSampleRate)
            : 44100
        filterMixRate = 1 / (sampleRate * 0.15)

        let rampDuration: Float = 1.0 // Full 0→1 sweep in 1 second
        return 1.0 / (sampleRate * rampDuration)
    }
}
