// Vector saves m, n, k, a_t, b_t, sparsity
std::vector<std::tuple<int, int, int, bool, bool, float>> inference_server_set = {
    std::make_tuple(7680, 1, 2560, false, false, 0.95),
    std::make_tuple(7860, 2, 2560, false, false, 0.95),
    std::make_tuple(7860, 4, 2560, false, false, 0.95),
    std::make_tuple(7680, 1500, 2560, false, false, 0.95),
    std::make_tuple(7860, 3000, 2560, false, false, 0.95),
    std::make_tuple(7860, 6000, 2560, false, false, 0.95),
    std::make_tuple(10752, 1, 3584, false, false, 0.95),
    std::make_tuple(10752, 2, 3584, false, false, 0.95),
    std::make_tuple(10752, 4, 3584, false, false, 0.95),
    std::make_tuple(10752, 1500, 3584, false, false, 0.95),
    std::make_tuple(10752, 3000, 3584, false, false, 0.95),
    std::make_tuple(10752, 6000, 3584, false, false, 0.95),
    std::make_tuple(7680, 1, 2560, false, false, 0.9),
    std::make_tuple(7860, 2, 2560, false, false, 0.9),
    std::make_tuple(7860, 4, 2560, false, false, 0.9),
    std::make_tuple(7680, 1500, 2560, false, false, 0.9),
    std::make_tuple(7860, 3000, 2560, false, false, 0.9),
    std::make_tuple(7860, 6000, 2560, false, false, 0.9),
    std::make_tuple(10752, 1, 3584, false, false, 0.9),
    std::make_tuple(10752, 2, 3584, false, false, 0.9),
    std::make_tuple(10752, 4, 3584, false, false, 0.9),
    std::make_tuple(10752, 1500, 3584, false, false, 0.9),
    std::make_tuple(10752, 3000, 3584, false, false, 0.9),
    std::make_tuple(10752, 6000, 3584, false, false, 0.9)
};

// Vector saves m, n, k, a_t, b_t, sparsity, data_type
std::vector<std::tuple<int, int, int, bool, bool, float>> inference_device_set = {
    std::make_tuple(7680, 1, 2560, false, false, 0.95),
    std::make_tuple(7680, 1500, 2560, false, false, 0.95),
    std::make_tuple(10752, 1, 3584, false, false, 0.95),
    std::make_tuple(10752, 1500, 3584, false, false, 0.95),
    std::make_tuple(7680, 1, 2560, false, false, 0.9),
    std::make_tuple(7680, 1500, 2560, false, false, 0.9),
    std::make_tuple(10752, 1, 3584, false, false, 0.9),
    std::make_tuple(10752, 1500, 3584, false, false, 0.9)
};

