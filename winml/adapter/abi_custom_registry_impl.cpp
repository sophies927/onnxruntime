// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "adapter/pch.h"

#ifdef USE_DML

#include "abi_custom_registry_impl.h"

namespace Windows::AI::MachineLearning::Adapter {

HRESULT STDMETHODCALLTYPE AbiCustomRegistryImpl::RegisterOperatorSetSchema(
  const MLOperatorSetId* opSetId,
  int baseline_version,
  const MLOperatorSchemaDescription* const* schema,
  uint32_t schemaCount,
  _In_opt_ IMLOperatorTypeInferrer* typeInferrer,
  _In_opt_ IMLOperatorShapeInferrer* shapeInferrer
) const noexcept try {
#ifdef LAYERING_DONE
  for (uint32_t i = 0; i < schemaCount; ++i) {
    telemetry_helper.RegisterOperatorSetSchema(
      schema[i]->name,
      schema[i]->inputCount,
      schema[i]->outputCount,
      schema[i]->typeConstraintCount,
      schema[i]->attributeCount,
      schema[i]->defaultAttributeCount
    );
  }
#endif

  // Delegate to base class
  return AbiCustomRegistry::RegisterOperatorSetSchema(
    opSetId, baseline_version, schema, schemaCount, typeInferrer, shapeInferrer
  );
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE AbiCustomRegistryImpl::RegisterOperatorKernel(
  const MLOperatorKernelDescription* opKernel,
  IMLOperatorKernelFactory* operatorKernelFactory,
  _In_opt_ IMLOperatorShapeInferrer* shapeInferrer
) const noexcept {
  return RegisterOperatorKernel(opKernel, operatorKernelFactory, shapeInferrer, nullptr, false, false, false);
}

HRESULT STDMETHODCALLTYPE AbiCustomRegistryImpl::RegisterOperatorKernel(
  const MLOperatorKernelDescription* opKernel,
  IMLOperatorKernelFactory* operatorKernelFactory,
  _In_opt_ IMLOperatorShapeInferrer* shapeInferrer,
  _In_opt_ IMLOperatorSupportQueryPrivate* supportQuery,
  bool isInternalOperator,
  bool supportsGraph,
  const uint32_t* requiredInputCountForGraph,
  _In_reads_(constantCpuInputCount) const uint32_t* requiredConstantCpuInputs,
  uint32_t constantCpuInputCount,
  _In_reads_(aliasCount) const std::pair<uint32_t, uint32_t>* aliases,
  uint32_t aliasCount
) const noexcept try {
#ifdef LAYERING_DONE
  // Log a custom op telemetry if the operator is not a built-in DML operator
  if (!isInternalOperator) {
    telemetry_helper.LogRegisterOperatorKernel(
      opKernel->name, opKernel->domain, static_cast<int>(opKernel->executionType)
    );
  }
#endif

  // Delegate to base class
  return AbiCustomRegistry::RegisterOperatorKernel(
    opKernel,
    operatorKernelFactory,
    shapeInferrer,
    supportQuery,
    isInternalOperator,
    supportsGraph,
    requiredInputCountForGraph,
    requiredConstantCpuInputs,
    constantCpuInputCount,
    aliases,
    aliasCount
  );
}
CATCH_RETURN();

}  // namespace Windows::AI::MachineLearning::Adapter

#endif USE_DML
