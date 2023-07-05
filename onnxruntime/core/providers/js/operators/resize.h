// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/tensor/upsamplebase.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Resize : public JsKernel, public UpsampleBase {
 public:
  Resize(const OpKernelInfo& info) : JsKernel(info), UpsampleBase(info) {
    auto resize_coordinate_transformation_mode = ResizeCoordinateTransformationModeToString(coordinate_transform_mode_);
    auto keep_aspect_ratio_policy = KeepAspectRatioPolicyToString(keep_aspect_ratio_policy_);
    auto nearest_mode = NearestModeToString(nearest_mode_);
    auto mode = UpsampleModeToString(mode_);
    std::vector<int32_t> axes;
    std::transform(axes_.begin(), axes_.end(), std::back_inserter(axes), [](auto& axis) { return gsl::narrow_cast<int32_t>(axis); });
    JSEP_INIT_KERNEL_ATTRIBUTE(Resize, ({
                                 "antialias" : $1,
                                 "axes" : $2 ? Array.from(HEAP32.subarray($3, $3 + $2)) : [],
                                 "coordinateTransformMode" : UTF8ToString($4),
                                 "cubicCoeffA" : $5,
                                 "excludeOutside" : $6,
                                 "extrapolationValue" : $7,
                                 "keepAspectRatioPolicy" : UTF8ToString($8),
                                 "mode" : UTF8ToString($9),
                                 "nearestMode" : UTF8ToString($10),
                               }),
                               static_cast<int32_t>(antialias_),
                               gsl::narrow_cast<int32_t>(axes.size()),
                               reinterpret_cast<int32_t>((axes.size() > 0) ? axes.data() : nullptr) >> 2,
                               resize_coordinate_transformation_mode.c_str(),
                               static_cast<double>(cubic_coeff_a_),
                               static_cast<int32_t>(exclude_outside_),
                               static_cast<double>(extrapolation_value_),
                               keep_aspect_ratio_policy.c_str(),
                               mode.c_str(),
                               nearest_mode.c_str());
  }

  std::string UpsampleModeToString(UpsampleMode mode) {
    switch (mode) {
      case UpsampleMode::NN:
        return UpsampleModeNN;
      case UpsampleMode::LINEAR:
        return UpsampleModeLinear;
      case UpsampleMode::CUBIC:
        return UpsampleModeCubic;
      default:
        ORT_THROW("UpsampleMode is not supported!");
    }
  }

  std::string KeepAspectRatioPolicyToString(AspectRatioPolicy policy) {
    switch (policy) {
      case AspectRatioPolicy::STRETCH:
        return "stretch";
      case AspectRatioPolicy::NOT_LARGER:
        return "not_larger";
      case AspectRatioPolicy::NOT_SMALLER:
        return "not_smaller";
      default:
        ORT_THROW("AspectRatioPolicy is not supported!");
    }
  }

  std::string ResizeCoordinateTransformationModeToString(const ResizeCoordinateTransformationMode mode) {
    switch (mode) {
      case ASYMMETRIC:
        return "asymmetric";
      case PYTORCH_HALF_PIXEL:
        return "pytorch_half_pixel";
      case TF_HALF_PIXEL_FOR_NN:
        return "tf_half_pixel_for_nn";
      case ALIGN_CORNERS:
        return "align_corners";
      case TF_CROP_AND_RESIZE:
        return "tf_crop_and_resize";
      case HALF_PIXEL:
        return "half_pixel";
      case HALF_PIXEL_SYMMETRIC:
        return "half_pixel_symmetric";
      default:
        ORT_THROW("ResizeCoordinateTransformationMode is not supported!");
    }
  }

  std::string NearestModeToString(const ResizeNearestMode mode) {
    switch (mode) {
      case ROUND_PREFER_FLOOR:
        return "round_prefer_floor";
      case ROUND_PREFER_CEIL:
        return "round_prefer_ceil";
      case FLOOR:
        return "floor";
      case CEIL:
        return "ceil";
      default:
        return "";
    }
  }
};

}  // namespace js
}  // namespace onnxruntime
