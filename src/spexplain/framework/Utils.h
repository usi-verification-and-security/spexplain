#ifndef SPEXPLAIN_FRAMEWORK_UTILS_H
#define SPEXPLAIN_FRAMEWORK_UTILS_H

#include "Framework.h"
#include "explanation/VarBound.h"

#include <spexplain/common/Interval.h>
#include <spexplain/common/Var.h>

#include <memory>

namespace spexplain {
std::unique_ptr<VarBound> intervalToOptVarBound(Framework const &, VarIdx, Interval const &);
} // namespace spexplain

#endif // SPEXPLAIN_FRAMEWORK_UTILS_H
