/**
 * Trading signals
 */

#pragma once

#include <string>
#include <unordered_map>
#include <cstdint>
#include <vector>

namespace trading_system {
namespace ml {

// Signal type
enum class SignalType {
    UNKNOWN,
    BUY,
    SELL,
    EXIT
};

// Pattern type
enum class PatternType {
    UNKNOWN,
    BULLISH_ENGULFING,
    BEARISH_ENGULFING,
    HAMMER,
    SHOOTING_STAR,
    MORNING_STAR,
    EVENING_STAR,
    DOJI,
    MARUBOZU,
    THREE_WHITE_SOLDIERS,
    THREE_BLACK_CROWS,
    BULLISH_HARAMI,
    BEARISH_HARAMI,
    PIERCING_LINE,
    DARK_CLOUD_COVER,
    BULLISH_KICKER,
    BEARISH_KICKER
};

// Trading signal
struct Signal {
    // Basic signal info
    std::string symbol;
    SignalType type;
    double confidence;
    double price;
    uint64_t timestamp;
    
    // Pattern information
    PatternType pattern;
    double pattern_confidence;
    
    // Technical indicators
    std::unordered_map<std::string, double> indicators;
    
    // Position sizing
    double position_size;
    double stop_loss;
    double take_profit;
    
    // Constructor
    Signal()
        : type(SignalType::UNKNOWN),
          confidence(0.0),
          price(0.0),
          timestamp(0),
          pattern(PatternType::UNKNOWN),
          pattern_confidence(0.0),
          position_size(0.0),
          stop_loss(0.0),
          take_profit(0.0) {
    }
    
    // Check if signal is valid
    bool isValid() const {
        return type != SignalType::UNKNOWN && confidence > 0.0 && price > 0.0;
    }
};

// Convert signal type to string
std::string signalTypeToString(SignalType type);

// Convert pattern type to string
std::string patternTypeToString(PatternType type);

// Convert string to signal type
SignalType stringToSignalType(const std::string& str);

// Convert string to pattern type
PatternType stringToPatternType(const std::string& str);

} // namespace ml
} // namespace trading_system