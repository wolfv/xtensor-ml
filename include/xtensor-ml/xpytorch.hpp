#include <torch/torch.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xeval.hpp>

namespace xt
{
    template <class T>
    auto map_to_torch_type()
    {
        if (std::is_same<T, double>::value) return torch::kFloat64;
        else if (std::is_same<T, float>::value) return torch::kFloat32;
        else if (std::is_same<T, int32_t>::value) return torch::kInt32;
        else if (std::is_same<T, int64_t>::value) return torch::kInt64;
        else if (std::is_same<T, int16_t>::value) return torch::kInt16;
        else if (std::is_same<T,  int8_t>::value) return torch::kInt8;
        else if (std::is_same<T, uint8_t>::value) return torch::kUInt8;
        throw std::runtime_error("Type not mapped to torch.");
    }

    template <class T, class S = std::vector<std::size_t>>
    auto to_xtensor(torch::Tensor& tensor)
    {
        return adapt(tensor.data<T>(), xtl::forward_sequence<S>(tensor.sizes()));
    }

    template <class T, class S = std::vector<std::size_t>>
    torch::Tensor from_xtensor(const xexpression<T>& expr)
    {
        auto options = torch::TensorOptions()
            .dtype(map_to_torch_type<typename T::value_type>())
            .layout(torch::kStrided)
        ;

        auto&& eval_expr = eval(expr.derived_cast());
        std::function<void(void*)> deleter = [](void* data) {};

        torch::ArrayRef<typename T::value_type> data_ref(eval_expr.data(), eval_expr.size());

        return torch::from_blob((void*) eval_expr.data(),
                                xtl::forward_sequence<std::vector<int64_t>>(eval_expr.shape()),
                                xtl::forward_sequence<std::vector<int64_t>>(eval_expr.strides()),
                                deleter,
                                options);
    }
}
