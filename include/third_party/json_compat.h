#pragma once

#include "simdjson.h"
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

// Compatibility layer to make simdjson API similar to nlohmann/json
namespace json_compat {

class json {
private:
    simdjson::dom::parser parser;
    simdjson::dom::element element;
    std::string raw_json;
    bool is_parsed = false;

public:
    // Default constructor
    json() : raw_json("null") {
        parse();
    }

    // Constructor from string
    json(const std::string& json_str) : raw_json(json_str) {
        parse();
    }
    
    // Constructor from various types
    json(int value) : raw_json(std::to_string(value)) {
        parse();
    }
    
    json(int64_t value) : raw_json(std::to_string(value)) {
        parse();
    }
    
    json(double value) : raw_json(std::to_string(value)) {
        parse();
    }
    
    json(bool value) : raw_json(value ? "true" : "false") {
        parse();
    }
    
    json(std::nullptr_t) : raw_json("null") {
        parse();
    }

    // Parse from string
    static json parse(const std::string& json_str) {
        return json(json_str);
    }

    // Access operators
    simdjson::dom::element operator[](const std::string& key) {
        if (!is_parsed) parse();
        simdjson::dom::element result;
        auto error = element.at_key(key).get(result);
        if (error) {
            // Create the key if it doesn't exist (for compatibility with nlohmann::json)
            // Note: This is a simplified implementation that doesn't actually modify the JSON.
            // In a real implementation, you would need to modify the underlying JSON.
            // TODO: Implement actual JSON modification for setting keys.
            throw std::runtime_error("JSON key not found: " + key);
        }
        return result;
    }
    
    // Const version of operator[]
    const simdjson::dom::element operator[](const std::string& key) const {
        if (!is_parsed) const_cast<json*>(this)->parse();
        simdjson::dom::element result;
        auto error = element.at_key(key).get(result);
        if (error) {
            throw std::runtime_error("JSON key not found: " + key);
        }
        return result;
    }

    simdjson::dom::element operator[](size_t index) {
        if (!is_parsed) parse();
        simdjson::dom::element result;
        auto error = element.at(index).get(result);
        if (error) {
            throw std::runtime_error("JSON index out of bounds: " + std::to_string(index));
        }
        return result;
    }

    // Type checking
    bool is_object() const {
        return element.type() == simdjson::dom::element_type::OBJECT;
    }

    bool is_array() const {
        return element.type() == simdjson::dom::element_type::ARRAY;
    }

    bool is_string() const {
        return element.type() == simdjson::dom::element_type::STRING;
    }

    bool is_number() const {
        return element.type() == simdjson::dom::element_type::DOUBLE ||
               element.type() == simdjson::dom::element_type::INT64 ||
               element.type() == simdjson::dom::element_type::UINT64;
    }

    bool is_bool() const {
        return element.type() == simdjson::dom::element_type::BOOL;
    }

    bool is_null() const {
        return element.type() == simdjson::dom::element_type::NULL_VALUE;
    }

    // Conversion functions
    template<typename T>
    T get() const {
        T result;
        auto error = element.get(result);
        if (error) {
            throw std::runtime_error("JSON type conversion error");
        }
        return result;
    }

    // Specialized getters for common types
    std::string get_string() const {
        std::string_view result;
        auto error = element.get_string().get(result);
        if (error) {
            throw std::runtime_error("JSON type conversion error: not a string");
        }
        return std::string(result);
    }

    int64_t get_int() const {
        int64_t result;
        auto error = element.get_int64().get(result);
        if (error) {
            throw std::runtime_error("JSON type conversion error: not an integer");
        }
        return result;
    }

    double get_double() const {
        double result;
        auto error = element.get_double().get(result);
        if (error) {
            throw std::runtime_error("JSON type conversion error: not a double");
        }
        return result;
    }

    bool get_bool() const {
        bool result;
        auto error = element.get_bool().get(result);
        if (error) {
            throw std::runtime_error("JSON type conversion error: not a boolean");
        }
        return result;
    }

    // Size methods for arrays and objects
    size_t size() const {
        if (is_array()) {
            simdjson::dom::array arr;
            auto error = element.get_array().get(arr);
            if (error) {
                throw std::runtime_error("JSON is not an array");
            }
            return arr.size();
        } else if (is_object()) {
            simdjson::dom::object obj;
            auto error = element.get_object().get(obj);
            if (error) {
                throw std::runtime_error("JSON is not an object");
            }
            return obj.size();
        }
        throw std::runtime_error("JSON element is neither array nor object");
    }

    bool empty() const {
        return size() == 0;
    }

    // Dump to string
    std::string dump() const {
        return raw_json;
    }

    // Pretty print with indentation
    std::string dump(int indent) const {
        // simdjson doesn't have built-in pretty printing, so we'll use a simple approach
        if (indent <= 0) {
            return dump();
        }
        
        // For simple pretty printing, we could use a third-party library or implement a basic version.
        // This is a placeholder that just returns the raw JSON.
        // In a real implementation, you would want to properly format the JSON.
        // TODO: Implement pretty printing for JSON output.
        return raw_json;
    }

    // Iteration support for objects
    class iterator {
    private:
        simdjson::dom::object::iterator it;
        simdjson::dom::object::iterator end_it;
        
    public:
        iterator(simdjson::dom::object::iterator begin, simdjson::dom::object::iterator end) 
            : it(begin), end_it(end) {}
        
        bool operator!=(const iterator& other) const {
            return it != other.it;
        }
        
        iterator& operator++() {
            ++it;
            return *this;
        }
        
        std::pair<std::string, simdjson::dom::element> operator*() const {
            // Explicitly create a std::pair from the key_value_pair
            return std::make_pair(std::string((*it).key), (*it).value);
        }
        
        std::string key() const {
            // Explicitly convert string_view to string
            return std::string((*it).key);
        }
        
        simdjson::dom::element value() const {
            return (*it).value;
        }
    };

    iterator begin() {
        simdjson::dom::object obj;
        auto error = element.get_object().get(obj);
        if (error) {
            throw std::runtime_error("JSON is not an object");
        }
        return iterator(obj.begin(), obj.end());
    }

    iterator end() {
        simdjson::dom::object obj;
        auto error = element.get_object().get(obj);
        if (error) {
            throw std::runtime_error("JSON is not an object");
        }
        return iterator(obj.end(), obj.end());
    }
    
    // Const versions of begin() and end()
    const iterator begin() const {
        simdjson::dom::object obj;
        auto error = element.get_object().get(obj);
        if (error) {
            throw std::runtime_error("JSON is not an object");
        }
        return iterator(obj.begin(), obj.end());
    }

    const iterator end() const {
        simdjson::dom::object obj;
        auto error = element.get_object().get(obj);
        if (error) {
            throw std::runtime_error("JSON is not an object");
        }
        return iterator(obj.end(), obj.end());
    }

    // Check if key exists
    bool contains(const std::string& key) const {
        simdjson::dom::element result;
        auto error = element.at_key(key).get(result);
        return !error;
    }

    // Array iteration support
    class array_iterator {
    private:
        simdjson::dom::array array;
        size_t index;
        size_t size;
        
    public:
        array_iterator(simdjson::dom::array arr, size_t idx)
            : array(arr), index(idx), size(arr.size()) {}
        
        bool operator!=(const array_iterator& other) const {
            return index != other.index;
        }
        
        array_iterator& operator++() {
            ++index;
            return *this;
        }
        
        simdjson::dom::element operator*() const {
            simdjson::dom::element result;
            auto error = array.at(index).get(result);
            if (error) {
                throw std::runtime_error("Array index out of bounds: " + std::to_string(index));
            }
            return result;
        }
    };

    array_iterator array_begin() {
        simdjson::dom::array arr;
        auto error = element.get_array().get(arr);
        if (error) {
            throw std::runtime_error("JSON is not an array");
        }
        return array_iterator(arr, 0);
    }

    array_iterator array_end() {
        simdjson::dom::array arr;
        auto error = element.get_array().get(arr);
        if (error) {
            throw std::runtime_error("JSON is not an array");
        }
        return array_iterator(arr, arr.size());
    }

    // Helper method to convert a simdjson element to a string
    static std::string element_to_string(const simdjson::dom::element& elem) {
        switch (elem.type()) {
            case simdjson::dom::element_type::ARRAY: {
                simdjson::dom::array arr;
                elem.get_array().get(arr);
                std::string result = "[";
                size_t i = 0;
                for (auto item : arr) {
                    if (i > 0) result += ",";
                    result += element_to_string(item);
                    i++;
                }
                result += "]";
                return result;
            }
            case simdjson::dom::element_type::OBJECT: {
                simdjson::dom::object obj;
                elem.get_object().get(obj);
                std::string result = "{";
                size_t i = 0;
                for (auto field : obj) {
                    if (i > 0) result += ",";
                    result += "\"" + std::string(field.key) + "\":" + element_to_string(field.value);
                    i++;
                }
                result += "}";
                return result;
            }
            case simdjson::dom::element_type::STRING: {
                std::string_view sv;
                elem.get_string().get(sv);
                return "\"" + std::string(sv) + "\"";
            }
            case simdjson::dom::element_type::INT64: {
                int64_t val;
                elem.get_int64().get(val);
                return std::to_string(val);
            }
            case simdjson::dom::element_type::UINT64: {
                uint64_t val;
                elem.get_uint64().get(val);
                return std::to_string(val);
            }
            case simdjson::dom::element_type::DOUBLE: {
                double val;
                elem.get_double().get(val);
                return std::to_string(val);
            }
            case simdjson::dom::element_type::BOOL: {
                bool val;
                elem.get_bool().get(val);
                return val ? "true" : "false";
            }
            case simdjson::dom::element_type::NULL_VALUE:
                return "null";
            default:
                return "null";
        }
    }

    // Methods to create or modify JSON objects
    void push_back(const simdjson::dom::element& value) {
        if (!is_array()) {
            throw std::runtime_error("Cannot push_back to a non-array JSON value");
        }
        
        // This is a simplified implementation that rebuilds the JSON.
        // In a real implementation, you would modify the underlying JSON more efficiently.
        // TODO: Implement efficient push_back for JSON arrays without rebuilding the entire JSON string.
        
        // Get the current array
        simdjson::dom::array arr;
        auto error = element.get_array().get(arr);
        if (error) {
            throw std::runtime_error("Failed to get array");
        }
        
        // Create a new JSON array with all existing elements plus the new one
        std::string new_json = "[";
        for (size_t i = 0; i < arr.size(); i++) {
            simdjson::dom::element elem;
            arr.at(i).get(elem);
            
            // This is a simplification - in reality you'd need to properly serialize each element
            if (i > 0) new_json += ",";
            new_json += element_to_string(elem);
        }
        
        // Add the new element
        if (arr.size() > 0) new_json += ",";
        new_json += element_to_string(value);
        new_json += "]";
        
        // Update the raw JSON and re-parse
        raw_json = new_json;
        parse();
    }
    
    // Set a key in an object
    void set(const std::string& key, const simdjson::dom::element& value) {
        if (!is_object()) {
            throw std::runtime_error("Cannot set key on a non-object JSON value");
        }
        
        // This is a simplified implementation that rebuilds the JSON.
        // In a real implementation, you would modify the underlying JSON more efficiently.
        // TODO: Implement efficient key setting for JSON objects without rebuilding the entire JSON string.
        
        // Get the current object
        simdjson::dom::object obj;
        auto error = element.get_object().get(obj);
        if (error) {
            throw std::runtime_error("Failed to get object");
        }
        
        // Create a new JSON object with all existing keys plus the new/updated one
        std::string new_json = "{";
        bool first = true;
        bool key_found = false;
        
        // Add all existing keys
        for (auto field : obj) {
            if (!first) new_json += ",";
            first = false;
            
            std::string field_key(field.key);
            if (field_key == key) {
                // Replace with new value
                new_json += "\"" + field_key + "\":" + element_to_string(value);
                key_found = true;
            } else {
                // Keep existing value
                new_json += "\"" + field_key + "\":" + element_to_string(field.value);
            }
        }
        
        // Add the new key if it wasn't found
        if (!key_found) {
            if (!first) new_json += ",";
            new_json += "\"" + key + "\":" + element_to_string(value);
        }
        
        new_json += "}";
        
        // Update the raw JSON and re-parse
        raw_json = new_json;
        parse();
    }
    
    // Erase a key from an object
    void erase(const std::string& key) {
        if (!is_object()) {
            throw std::runtime_error("Cannot erase key from a non-object JSON value");
        }
        
        // This is a simplified implementation that rebuilds the JSON.
        // In a real implementation, you would modify the underlying JSON more efficiently.
        // TODO: Implement efficient key erasure for JSON objects without rebuilding the entire JSON string.
        
        // Get the current object
        simdjson::dom::object obj;
        auto error = element.get_object().get(obj);
        if (error) {
            throw std::runtime_error("Failed to get object");
        }
        
        // Create a new JSON object without the erased key
        std::string new_json = "{";
        bool first = true;
        
        // Add all keys except the one to erase
        for (auto field : obj) {
            std::string field_key(field.key);
            if (field_key != key) {
                if (!first) new_json += ",";
                first = false;
                new_json += "\"" + field_key + "\":" + element_to_string(field.value);
            }
        }
        
        new_json += "}";
        
        // Update the raw JSON and re-parse
        raw_json = new_json;
        parse();
    }

    // Comparison operators
    bool operator==(const json& other) const {
        // This is a simplified implementation.
        // In a real implementation, you would compare the actual JSON values.
        // TODO: Implement proper JSON value comparison.
        return raw_json == other.raw_json;
    }

    bool operator!=(const json& other) const {
        return !(*this == other);
    }

    // Create a JSON value from a C++ value
    template<typename T>
    static json from_value(const T& value) {
        return json(value);
    }
    
    // Merge two JSON objects
    void merge_patch(const json& other) {
        if (!is_object() || !other.is_object()) {
            throw std::runtime_error("merge_patch requires both JSONs to be objects");
        }
        
        // Iterate through the other object and apply its keys
        for (auto it = other.begin(); it != other.end(); ++it) {
            std::string key = it.key();
            simdjson::dom::element value = it.value();
            set(key, value);
        }
    }

private:
    void parse() {
        try {
            auto error = parser.parse(raw_json).get(element);
            if (error) {
                throw std::runtime_error("JSON parse error: " + std::string(simdjson::error_message(error)));
            }
            is_parsed = true;
        } catch (const simdjson::simdjson_error& e) {
            throw std::runtime_error("JSON parse error: " + std::string(e.what()));
        }
    }
};

// Template specializations for common types
template<>
inline std::string json::get<std::string>() const {
    return get_string();
}

template<>
inline int json::get<int>() const {
    return static_cast<int>(get_int());
}

template<>
inline int64_t json::get<int64_t>() const {
    return get_int();
}

template<>
inline double json::get<double>() const {
    return get_double();
}

template<>
inline bool json::get<bool>() const {
    return get_bool();
}

// Helper functions to create JSON objects
inline json make_object() {
    return json("{}");
}

inline json make_array() {
    return json("[]");
}

// Helper function to create a JSON object from key-value pairs
template<typename... Args>
inline json object(Args&&... args) {
json obj = make_object();
// Implementation would depend on how you want to pass key-value pairs.
// This is a placeholder.
// TODO: Implement JSON object creation from key-value pairs.
return obj;
}

// Helper function to create a JSON array from values
template<typename... Args>
inline json array(Args&&... args) {
json arr = make_array();
// Implementation would depend on how you want to pass values.
// This is a placeholder.
// TODO: Implement JSON array creation from values.
return arr;
}

// Function to parse JSON from string (static method already exists in the class)
inline json parse(const std::string& json_str) {
    return json::parse(json_str);
}

} // namespace json_compat

// For backward compatibility
using json = json_compat::json;

// Additional compatibility functions
namespace std {
    // This allows json objects to be used in unordered containers
    template<>
    struct hash<json_compat::json> {
        size_t operator()(const json_compat::json& j) const {
            return std::hash<std::string>{}(j.dump());
        }
    };
}