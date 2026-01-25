#include "hamming_dist_top_k_tiling.h"
#include "hamming_dist_top_k.h"
#include "hamming_dist_top_k_split.h"
#include "register/op_def_registry.h"
// #include "context_util.h"
#include "register/op_impl_registry.h"


namespace optiling {
static ge::graphStatus TilingPrepareForHammingDistTopK(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    std::cout << "enter TilingFunc\n";
    ge::graphStatus ret;
    HammingDistTopKSplitSTiling hammingDistTopKSplitSTiling(context);
    hammingDistTopKSplitSTiling.GetShapeAttrsInfo();
    hammingDistTopKSplitSTiling.GetPlatformInfo();
    auto can_split = hammingDistTopKSplitSTiling.IsCapable();
    if (can_split) {
        hammingDistTopKSplitSTiling.DoOpTiling();
        hammingDistTopKSplitSTiling.DoLibApiTiling();
        hammingDistTopKSplitSTiling.GetWorkspaceSize();
        hammingDistTopKSplitSTiling.PostTiling();
        context->SetTilingKey(hammingDistTopKSplitSTiling.GetTilingKey());
        std::cout << "enter TilingFunc split SetTilingKey\n";
        hammingDistTopKSplitSTiling.PrintTilingData(); //print for debug by ldeng
        return ge::GRAPH_SUCCESS;
    }
    
    HammingDistTopKTiling hammingDistTopKTiling(context);
    hammingDistTopKTiling.GetShapeAttrsInfo();
    hammingDistTopKTiling.GetPlatformInfo();
    hammingDistTopKTiling.IsCapable();
    hammingDistTopKTiling.DoOpTiling();
    hammingDistTopKTiling.DoLibApiTiling();
    hammingDistTopKTiling.GetWorkspaceSize();
    hammingDistTopKTiling.PostTiling();
    context->SetTilingKey(hammingDistTopKTiling.GetTilingKey());
    std::cout << "enter TilingFunc SetTilingKey\n";
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HammingDistTopK)
    .Tiling(TilingFunc)
    .TilingParse<HammingDistTopKCompileInfo>(TilingPrepareForHammingDistTopK);
}
