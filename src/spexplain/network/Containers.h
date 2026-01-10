#ifndef SPEXPLAIN_NETWORK_CONTAINERS_H
#define SPEXPLAIN_NETWORK_CONTAINERS_H

#include "Network.h"

#include <spexplain/common/Macro.h>

#include <cassert>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace spexplain {

template<typename T>
class NetworkMap {
public:
    using MappedType = T;

    NetworkMap() = default;
    inline NetworkMap(Network const &);
    NetworkMap(NetworkMap const &) = default;
    NetworkMap & operator=(NetworkMap const &) = default;
    NetworkMap(NetworkMap &&) = default;
    NetworkMap & operator=(NetworkMap &&) = default;

    inline void init(Network const &) noexcept;

    inline void setNetwork(Network const &) noexcept;

    inline void clear() noexcept;

    inline bool empty() const noexcept;

    inline bool tryEmplace(std::size_t layer, std::size_t node, auto &&... args);
    inline bool insertOrAssign(std::size_t layer, std::size_t node, auto &&... args);

    inline bool contains(std::size_t layer, std::size_t node) const;
    inline T const & getAt(std::size_t layer, std::size_t node) const;
    inline T & getAt(std::size_t layer, std::size_t node);
    inline std::optional<T const *> tryGetPtrAt(std::size_t layer, std::size_t node) const;
    inline std::optional<T *> tryGetPtrAt(std::size_t layer, std::size_t node);
    inline std::optional<T> tryGetAt(std::size_t layer, std::size_t node) const;

private:
    template<bool overwriteV>
    inline bool insertImpl(std::size_t layer, std::size_t node, auto &&... args);

    Network const * networkPtr{};

    std::vector<std::unordered_map<std::size_t, T>> vectorOfMaps{};
};

template<typename T>
NetworkMap<T>::NetworkMap(Network const & nw) {
    setNetwork(nw);
}

template<typename T>
void NetworkMap<T>::init(Network const & nw) noexcept {
    setNetwork(nw);
    clear();
}

template<typename T>
void NetworkMap<T>::setNetwork(Network const & nw) noexcept {
    networkPtr = &nw;
}

template<typename T>
void NetworkMap<T>::clear() noexcept {
    vectorOfMaps.clear();
}

template<typename T>
bool NetworkMap<T>::empty() const noexcept {
    return vectorOfMaps.empty();
}

template<typename T>
bool NetworkMap<T>::tryEmplace(std::size_t layer, std::size_t node, auto &&... args) {
    return insertImpl<false>(layer, node, FORWARD(args)...);
}

template<typename T>
bool NetworkMap<T>::insertOrAssign(std::size_t layer, std::size_t node, auto &&... args) {
    return insertImpl<true>(layer, node, FORWARD(args)...);
}

template<typename T>
template<bool overwriteV>
bool NetworkMap<T>::insertImpl(std::size_t layer, std::size_t node, auto &&... args) {
    using namespace std::string_literals;

    assert(networkPtr);
    auto const nHiddenLayers = networkPtr->nHiddenLayers();
    auto const layerSize = networkPtr->getLayerSize(layer);

    if (layer > nHiddenLayers) {
        throw std::out_of_range{"Hidden layer is out of range: "s + std::to_string(layer) + " > " +
                                std::to_string(nHiddenLayers)};
    }
    if (node >= layerSize) {
        throw std::out_of_range{"Node is out of range: "s + std::to_string(node) + " >= " + std::to_string(layerSize)};
    }

    vectorOfMaps.resize(nHiddenLayers + 1);
    auto & map = vectorOfMaps[layer];
    map.reserve(layerSize);
    if constexpr (overwriteV) {
        map.insert_or_assign(node, T{FORWARD(args)...});
        return true;
    } else {
        auto [it, inserted] = map.try_emplace(node, FORWARD(args)...);
        return inserted;
    }
}

template<typename T>
bool NetworkMap<T>::contains(std::size_t layer, std::size_t node) const {
    if (vectorOfMaps.size() <= layer) { return false; }

    auto const & map = vectorOfMaps[layer];
    return map.contains(node);
}

template<typename T>
T const & NetworkMap<T>::getAt(std::size_t layer, std::size_t node) const {
    auto const & map = vectorOfMaps.at(layer);
    return map.at(node);
}

template<typename T>
T & NetworkMap<T>::getAt(std::size_t layer, std::size_t node) {
    return const_cast<T *>(std::as_const(*this).getAt(layer, node));
}

template<typename T>
std::optional<T const *> NetworkMap<T>::tryGetPtrAt(std::size_t layer, std::size_t node) const {
    if (vectorOfMaps.size() <= layer) { return std::nullopt; }

    auto const & map = vectorOfMaps[layer];
    if (auto it = map.find(node); it != map.end()) { return &it->second; }

    return std::nullopt;
}

template<typename T>
std::optional<T *> NetworkMap<T>::tryGetPtrAt(std::size_t layer, std::size_t node) {
    return const_cast<T *>(std::as_const(*this).tryGetPtrAt(layer, node));
}

template<typename T>
std::optional<T> NetworkMap<T>::tryGetAt(std::size_t layer, std::size_t node) const {
    if (vectorOfMaps.size() <= layer) { return std::nullopt; }

    auto const & map = vectorOfMaps[layer];
    if (auto it = map.find(node); it != map.end()) { return it->second; }

    return std::nullopt;
}
} // namespace spexplain

#endif // SPEXPLAIN_NETWORK_CONTAINERS_H
