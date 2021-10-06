// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Holds all location information of initializers defined in all graphs (main graph + all subgraphs).
// Currently does not handle "shadow" initializers.
struct GraphInitializersLocationInfo {
  void AccumulateInitializerLocationsInSubgraphs(Graph& graph,
                                                 const InitializedTensorSet& initializers,
                                                 const KernelRegistryManager& kernel_registries,
                                                 /*out*/ std::unordered_map<std::string, std::unordered_set<int>>& initializer_to_location_map) const;

  void MakeProviderInitializerDuplicates(Graph& graph, const KernelRegistryManager& kernel_registries) const;

  // Names of initializers consumed by non-provider nodes
  // Names in here can't be in provider_initializer_names_
  mutable std::unordered_set<std::string> non_provider_initializer_names_;

  // Names of initializers consumed by provider nodes
  // Names in here can't be in non_provider_initializer_names_
  mutable std::unordered_set<std::string> provider_initializer_names_;

  // Name of the "dupe" initializer for cases where initializers are consumed on both provider
  // and non-provider nodes.
  // Keys for this map will be found in non_provider_initializer_names_.
  // Not all entries in non_provider_initializer_names_ will be found in this map.
  mutable std::unordered_map<std::string, std::string> non_provider_initializer_names_to_provider_dupe_initializer_names_;
};

/**
@Class MemcpyTransformer

Transformer that inserts nodes to copy memory between devices when needed.
*/
class MemcpyTransformer : public GraphTransformer {
 public:
  MemcpyTransformer(const std::vector<std::string>& provider_types, const KernelRegistryManager& registry_manager)
      : GraphTransformer("MemcpyTransformer"), provider_types_(provider_types), registry_manager_(std::cref(registry_manager)) {}

 private:
  common::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  const std::vector<std::string> provider_types_;
  std::reference_wrapper<const KernelRegistryManager> registry_manager_;
  GraphInitializersLocationInfo graph_initializers_location_info_;
};

}  // namespace onnxruntime
