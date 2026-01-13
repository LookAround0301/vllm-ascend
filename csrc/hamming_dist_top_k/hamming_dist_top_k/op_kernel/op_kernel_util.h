/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file op_kernel_util.h
 * \brief
 */

#ifndef OP_KERNEL_UTIL_H_
#define OP_KERNEL_UTIL_H_

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
// #include "error/ops_error.h"

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
// #include "tiling/tiling_api.h"

// namespace optiling {
// BEGIN_TILING_DATA_DEF(HammingDistTopKTilingParams)
//     TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
//     TILING_DATA_FIELD_DEF(uint32_t, batch);
// 	TILING_DATA_FIELD_DEF(uint32_t, batchN);
//     TILING_DATA_FIELD_DEF(uint32_t, head);
//     TILING_DATA_FIELD_DEF(uint32_t, dimension);
//     TILING_DATA_FIELD_DEF(uint32_t, reducedBatch);
// 	TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
//     TILING_DATA_FIELD_DEF(uint32_t, sink);
//     TILING_DATA_FIELD_DEF(uint32_t, recent);
//     TILING_DATA_FIELD_DEF(bool, supportOffload);
//     TILING_DATA_FIELD_DEF(uint32_t, layerSize);
//     TILING_DATA_FIELD_DEF(uint32_t, matmulResultSize);
//     TILING_DATA_FIELD_DEF(uint32_t, topKValueSize);
//     TILING_DATA_FIELD_DEF(uint32_t, topKIdexSize);
//     TILING_DATA_FIELD_DEF(uint32_t, topKInnerSize);
//     TILING_DATA_FIELD_DEF(uint32_t, maxK);
//     TILING_DATA_FIELD_DEF(uint32_t, tileN1);
//     TILING_DATA_FIELD_DEF(uint32_t, sBlockSize);
//     TILING_DATA_FIELD_DEF(uint32_t, blockCount);
//     TILING_DATA_FIELD_DEF(uint32_t, tileN3);
//     TILING_DATA_FIELD_DEF(uint32_t, tileN2);
//     TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatch);
//     TILING_DATA_FIELD_DEF(uint32_t, singleCoreSeqLen);
//     TILING_DATA_FIELD_DEF(uint32_t, outter);
//     TILING_DATA_FIELD_DEF(uint32_t, inner);
//     TILING_DATA_FIELD_DEF(uint32_t, topkN);
//     TILING_DATA_FIELD_DEF(uint64_t, mmGmOffset);
//     TILING_DATA_FIELD_DEF(uint32_t, qHead);
//     TILING_DATA_FIELD_DEF(uint64_t, qUnpackGmOffset);
//     TILING_DATA_FIELD_DEF(uint32_t, headGroupNum);
// END_TILING_DATA_DEF;

// REGISTER_TILING_DATA_CLASS(HammingDistTopKTilingParamsOp, HammingDistTopKTilingParams)

// BEGIN_TILING_DATA_DEF(HammingDistTopKTilingData)
//     TILING_DATA_FIELD_DEF_STRUCT(HammingDistTopKTilingParams, params);
//     TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
//     TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTiling);
// END_TILING_DATA_DEF;

// REGISTER_TILING_DATA_CLASS(HammingDistTopK, HammingDistTopKTilingData)
// REGISTER_TILING_DATA_CLASS(HammingDistTopKTilingDataOp, HammingDistTopKTilingData)

// enum class MatmulConfig {
//     NULL_CONFIG = 0,
//     NORMAL_CONFIG = 1,
//     MDL_CONFIG = 2
// };


// }
// DCauto1!2@
#endif
