#ifndef SPEXPLAIN_MACRO_H
#define SPEXPLAIN_MACRO_H

#include <memory>
#include <type_traits>
#include <utility>

#define FORWARD(arg) std::forward<decltype(arg)>(arg)

#define REMOVE_CVREF(obj) std::remove_cvref_t<decltype(obj)>

#define MAKE_UNIQUE(obj) std::make_unique<REMOVE_CVREF(obj)>(obj)
#define MAKE_SHARED(obj) std::make_shared<REMOVE_CVREF(obj)>(obj)

#endif // SPEXPLAIN_MACRO_H
